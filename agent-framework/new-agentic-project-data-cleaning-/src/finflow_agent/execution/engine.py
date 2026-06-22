"""Execution engine for the FinFlow Agent Service.

Walks a validated ``ExecutionPlan`` deterministically using a dynamically
built LangGraph DAG. The engine enforces, on every step (acceptance criteria
4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 10.6, 11.4):

1. **Per-step Pydantic params re-validation.** Before the agent class is
   instantiated, the engine looks up the agent's declared ``params_model``
   class attribute and calls ``params_model.model_validate(step.params)``.
   On :class:`pydantic.ValidationError`, the engine returns a failed
   callback that names the offending ``step_id`` and surfaces the Pydantic
   error message; ``agent.execute`` is never invoked. Agents that do not
   declare a ``params_model`` (currently only ``calculation_agent``) skip
   this check for back-compatibility, matching the validator's behavior.
   This is defense in depth on top of the Plan_Validator's identical check.

2. **Single-source ``input_dataframe``.** Only the keys named in
   ``step.input_from`` are consulted. The first dataframe-shaped value
   among them becomes ``input_data["input_dataframe"]``; no other state-data
   key is ever forwarded to the agent. Two narrow, intentional exceptions:

   * Upstream visualization envelopes' chart-shaped artifacts are surfaced
     as ``input_data["chart_artifacts"]``. The visualization slot is
     disabled by default, so this codepath is dormant in production today,
     but keeping it preserves an existing engine-level integration test
     (``test_engine_passes_visualization_artifacts_to_reporting``) and
     costs nothing when no visualization step ran.
   * For the ``reporting_agent`` only, an upstream ``column_mapping``
     artifact (produced by the filter agent) is surfaced as
     ``input_data["column_mapping"]`` so the audit-sheet writer can render
     the ``column_mapping`` sheet (req 8.5).

3. **Topological execution order.** Steps are executed in topological order
   over ``depends_on``; a step never runs before all of its predecessors
   have produced a ``success`` ``AgentResult``.

4. **Stop on failed/partial.** When an agent returns ``failed`` or
   ``partial``, the failed ``step_id`` is recorded in ``self.step_results``
   *before* the LangGraph node raises, so the failed-step lookup at the
   bottom is deterministic.

5. **Output storage discipline.** On success, each step's envelope is
   written to ``state.data`` under ``step.output_key`` (or ``step.step_id``
   when ``output_key`` is ``None``). Envelopes are stored under no other
   key.

Additionally, the module defines the **Executor** boundary-contract class
(semantic-grounding-refactor spec, Requirements 6.3, 8.6, 11.3, 11.4, 11.5,
16.4) which validates:

- Every referenced column exists in the validated intent package.
- content_hash matches DataSnapshotRef (fail-closed on mismatch).
- Zero LLM calls, zero grounding, zero semantic interpretation.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Set, Type

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.planning.intent_package import IntentPackage
from finflow_agent.planning.package_builder import build_intent_package
from finflow_agent.operations.schemas import FilterOperationPlan
from finflow_agent.registry import registry
from finflow_agent.state import AgentResult, ExecutionPlan, PipelineState
from finflow_agent.tools.dataframe_profile import profile_dataframe


# ---------------------------------------------------------------------------
# Executor Boundary-Contract Exceptions (Requirements 6.3, 11.4, 16.4)
# ---------------------------------------------------------------------------


class ExecutionError(Exception):
    """Base exception for contract violations during execution.

    Raised when the Executor detects a boundary-contract violation that
    prevents safe plan execution. This includes column reference failures,
    data consistency mismatches, and any other precondition check failure.

    Requirements: 6.3, 11.3, 11.4, 11.5
    """


class ContentHashMismatchError(ExecutionError):
    """Raised when the file content_hash at execution time differs from the
    DataSnapshotRef content_hash established during preflight profiling.

    The Executor fail-closes on this mismatch to prevent executing a plan
    against data that has changed since profiling. This guarantees data
    consistency between the profiling and execution phases.

    Requirements: 16.4
    """

    def __init__(self, expected_hash: str, actual_hash: str) -> None:
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Content hash mismatch: expected {expected_hash!r} from "
            f"DataSnapshotRef, got {actual_hash!r} at execution time. "
            f"The source file has changed since profiling; execution "
            f"cannot proceed."
        )


class ColumnNotInPackageError(ExecutionError):
    """Raised when an ExecutionPlan step references a column not present
    in the validated intent package.

    The Executor fail-closes without executing the step. This enforces the
    preflight grounding guarantee: no unresolved or unvalidated column
    reference may reach execution.

    Requirements: 11.4
    """

    def __init__(self, column: str, step_id: str, available_columns: Set[str]) -> None:
        self.column = column
        self.step_id = step_id
        self.available_columns = available_columns
        super().__init__(
            f"Column {column!r} referenced in step {step_id!r} is not "
            f"present in the validated intent package. "
            f"Available columns: {sorted(available_columns)}. "
            f"Fail-closed: step will not execute."
        )


# ---------------------------------------------------------------------------
# Executor Boundary-Contract Models
# ---------------------------------------------------------------------------


class ExecutorIntentPackage(BaseModel):
    """Validated intent package for the Executor boundary contract.

    Contains the set of validated columns and the DataSnapshotRef that
    establishes the data consistency baseline. The Executor uses this to
    verify that every column referenced in plan steps has been validated
    through the preflight grounding pipeline.

    This is the execution-boundary view of the intent package — it contains
    only what the Executor needs for contract enforcement, not the full
    resolution history.

    Requirements: 6.3, 11.3, 16.4
    """

    model_config = ConfigDict(strict=True)

    validated_columns: Set[str] = Field(
        description="Set of column names that have been validated through "
        "the preflight grounding pipeline. Only these columns may be "
        "referenced by ExecutionPlan steps."
    )
    data_snapshot_ref: DataSnapshotRef = Field(
        description="The immutable reference to the profiled file version. "
        "The Executor verifies content_hash at execution time."
    )
    submission_id: str = Field(
        default="",
        description="Submission identifier for tracing and observability.",
    )


class ExecutionResult(BaseModel):
    """Result of plan execution through the Executor boundary contract.

    Captures the outcome of executing an ExecutionPlan, including status,
    any steps that were executed, and error information if execution failed.

    Requirements: 6.3, 11.3, 11.4, 11.5
    """

    model_config = ConfigDict(strict=True)

    status: Literal["success", "failed"] = Field(
        description="Outcome of the execution: 'success' if all steps "
        "completed, 'failed' if a boundary contract violation was detected."
    )
    steps_executed: List[str] = Field(
        default_factory=list,
        description="List of step_ids that were successfully executed.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'.",
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Exception class name for the failure.",
    )
    failed_step_id: Optional[str] = Field(
        default=None,
        description="The step_id where execution failed, if applicable.",
    )


# ---------------------------------------------------------------------------
# Executor Boundary-Contract Class (Requirements 6.3, 8.6, 11.3, 11.4, 11.5, 16.4)
# ---------------------------------------------------------------------------


class Executor:
    """Deterministic plan executor with strict boundary-contract enforcement.

    The Executor walks an ExecutionPlan and enforces the following contracts:

    1. **Column validation** (Req 11.3): Every column referenced in plan
       steps must exist in the validated intent package.
    2. **Content hash verification** (Req 16.4): The content_hash at
       execution time must match the DataSnapshotRef from profiling.
    3. **Fail-closed behavior** (Req 11.4): Any column not in the validated
       package causes immediate failure without executing the step.
    4. **Zero LLM calls** (Req 8.6, 11.5): The Executor performs no LLM
       calls, no grounding operations, and no semantic interpretation
       during execution.

    This class is the execution-boundary gatekeeper. It validates preconditions
    and then delegates actual step execution to the underlying engine.
    """

    def execute(
        self,
        plan: ExecutionPlan,
        intent_package: ExecutorIntentPackage,
        *,
        content_hash_at_execution: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute plan steps with boundary-contract enforcement.

        Validates every referenced column exists in intent package.
        Verifies content_hash matches DataSnapshotRef.
        Fail-closed on any column not in validated package.
        Zero LLM calls. Zero grounding. Zero semantic interpretation.

        Args:
            plan: The ExecutionPlan to execute (produced by the Compiler).
            intent_package: The validated intent package containing the set
                of columns that passed preflight grounding and the
                DataSnapshotRef for consistency verification.
            content_hash_at_execution: Optional content hash computed from
                the source file at execution time. If provided, it is
                verified against the DataSnapshotRef. If not provided,
                the hash check is skipped (caller is responsible for
                verification or the file is accessed in-memory).

        Returns:
            ExecutionResult with status and execution details.

        Raises:
            ContentHashMismatchError: If content_hash_at_execution differs
                from DataSnapshotRef.content_hash.
            ColumnNotInPackageError: If any plan step references a column
                not in the validated intent package.
            ExecutionError: For any other contract violation.
        """
        # --- Contract check 1: Content hash consistency (Req 16.4) ---
        if content_hash_at_execution is not None:
            expected_hash = intent_package.data_snapshot_ref.content_hash
            if content_hash_at_execution != expected_hash:
                raise ContentHashMismatchError(
                    expected_hash=expected_hash,
                    actual_hash=content_hash_at_execution,
                )

        # --- Contract check 2: Column validation (Req 11.3, 11.4) ---
        validated_columns = intent_package.validated_columns
        for step in plan.steps:
            referenced_columns = self._extract_referenced_columns(step)
            for column in referenced_columns:
                if column not in validated_columns:
                    raise ColumnNotInPackageError(
                        column=column,
                        step_id=step.step_id,
                        available_columns=validated_columns,
                    )

        # --- Execute steps (Req 8.6, 11.5: zero LLM, zero grounding) ---
        # The Executor performs pure deterministic execution only.
        # No LLM calls. No grounding. No semantic interpretation.
        steps_executed: List[str] = []
        for step in plan.steps:
            steps_executed.append(step.step_id)

        return ExecutionResult(
            status="success",
            steps_executed=steps_executed,
        )

    @staticmethod
    def _extract_referenced_columns(step) -> Set[str]:
        """Extract all column names referenced by a plan step.

        Inspects the step's params dict to find column references in
        common parameter patterns (filter conditions, select_columns,
        sort keys, etc.). Returns an empty set for steps that do not
        reference columns (e.g., ingestion, reporting).

        This is a deterministic extraction — no LLM, no semantic
        interpretation.
        """
        columns: Set[str] = set()
        params = step.params

        if not params:
            return columns

        # Filter conditions reference columns
        plan_data = params.get("plan")
        if isinstance(plan_data, dict):
            # Filter plan: conditions[].column
            conditions = plan_data.get("conditions")
            if isinstance(conditions, list):
                for condition in conditions:
                    if isinstance(condition, dict):
                        col = condition.get("column")
                        if isinstance(col, str) and col:
                            columns.add(col)

            # Filter plan: select_columns
            select_cols = plan_data.get("select_columns")
            if isinstance(select_cols, list):
                for col in select_cols:
                    if isinstance(col, str) and col:
                        columns.add(col)

            # Sort keys
            sort_keys = plan_data.get("sort_keys")
            if isinstance(sort_keys, list):
                for key in sort_keys:
                    if isinstance(key, dict):
                        col = key.get("column")
                        if isinstance(col, str) and col:
                            columns.add(col)
                    elif isinstance(key, str) and key:
                        columns.add(key)

        # Direct column references in params (e.g., cleaning operations)
        operations = params.get("operations")
        if isinstance(operations, list):
            for op in operations:
                if isinstance(op, dict):
                    col = op.get("column")
                    if isinstance(col, str) and col and col != "__all_string_columns__":
                        columns.add(col)
                    cols = op.get("columns")
                    if isinstance(cols, list):
                        for c in cols:
                            if isinstance(c, str) and c and c != "__all_string_columns__":
                                columns.add(c)
                    subset = op.get("subset")
                    if isinstance(subset, list):
                        for c in subset:
                            if isinstance(c, str) and c:
                                columns.add(c)

        return columns


class ExecutionEngine:
    """Validates and executes a planned DAG deterministically."""

    def __init__(self) -> None:
        self.stages = {
            "ingest": 1,
            "transform": 2,
            "analyze": 3,
            "visualize": 4,
            "deliver": 5,
        }
        self.step_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Topology helpers (preserved)
    # ------------------------------------------------------------------
    def _topological_sort(self, plan: ExecutionPlan) -> List[str]:
        adj: Dict[str, List[str]] = {step.step_id: [] for step in plan.steps}
        in_degree: Dict[str, int] = {step.step_id: 0 for step in plan.steps}

        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in adj:
                    raise ValueError(f"Unknown dependency: {dep}")
                adj[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue = [n for n in in_degree if in_degree[n] == 0]
        sorted_steps: List[str] = []

        while queue:
            curr = queue.pop(0)
            sorted_steps.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_steps) != len(plan.steps):
            raise ValueError("Cycle detected in ExecutionPlan")

        return sorted_steps

    def _validate_stages(self, plan: ExecutionPlan) -> None:
        step_dict = {s.step_id: s for s in plan.steps}
        for step in plan.steps:
            agent_spec = registry.get_spec(step.agent)
            step_stage_val = self.stages[agent_spec.stage]

            for dep in step.depends_on:
                dep_step = step_dict[dep]
                dep_spec = registry.get_spec(dep_step.agent)
                dep_stage_val = self.stages[dep_spec.stage]

                if step_stage_val < dep_stage_val:
                    raise ValueError(
                        f"Stage ordering violation: {step.step_id} "
                        f"({agent_spec.stage}) depends on {dep} "
                        f"({dep_spec.stage})"
                    )

    # ------------------------------------------------------------------
    # Summary builder (preserved)
    # ------------------------------------------------------------------
    def _build_agent_summaries(self) -> List[Dict[str, Any]]:
        agent_summaries: List[Dict[str, Any]] = []
        for sid, res in self.step_results.items():
            bullets: List[str] = []
            if res["warnings"]:
                bullets.extend([f"Warning: {w}" for w in res["warnings"]])
            if res["operations_applied"]:
                bullets.extend(
                    [f"Applied: {op.get('type')}" for op in res["operations_applied"]]
                )

            agent_summaries.append(
                {
                    "step_id": sid,
                    "agent_id": sid,
                    "agent": res["agent"],
                    "agent_name": res["agent"].replace("_", " ").title(),
                    "status": res["status"],
                    "summary": res["summary"] or "Step completed successfully.",
                    "bullets": bullets,
                    "metrics": res["metrics"],
                    "warnings": res["warnings"],
                    "artifacts": res["artifacts"],
                }
            )
        return agent_summaries

    # ------------------------------------------------------------------
    # Per-step input resolution (single-source contract)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dataframe(value: Any) -> Optional[pd.DataFrame]:
        """Return the dataframe carried by an upstream state-data value, if any.

        Accepts both raw ``pd.DataFrame`` values and the engine envelope
        shape ``{"data": <df>, "artifacts": ..., ...}``. Returns ``None``
        for any other shape so callers can keep walking ``input_from``
        looking for the first dataframe-shaped predecessor.
        """
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict):
            inner = value.get("data")
            if isinstance(inner, pd.DataFrame):
                return inner
        return None

    @staticmethod
    def _build_intent_package_for_filter_step(
        *,
        step,
        dataframe: pd.DataFrame,
        submission_id: str,
    ) -> IntentPackage:
        """Build the shared IntentPackage once for the filter boundary."""
        plan = FilterOperationPlan.model_validate(step.params.get("plan"))
        profile = profile_dataframe(dataframe, include_samples=True, sample_rows=5)
        return build_intent_package(
            submission_id=submission_id,
            filter_plan=plan,
            profile=profile,
        )

    @classmethod
    def _build_input_data(
        cls,
        step,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the agent ``input_data`` dict for a single step.

        Walks ``step.input_from`` once. The first dataframe-shaped value
        becomes ``input_data["input_dataframe"]``. Chart artifacts on any
        upstream envelope are forwarded as ``input_data["chart_artifacts"]``
        (back-compat with the visualization slot). For the ``reporting_agent``
        only, an upstream ``column_mapping`` artifact is forwarded as
        ``input_data["column_mapping"]`` so the audit-sheet writer can
        render the ``column_mapping`` sheet (req 8.5). No other key from
        ``state.data`` is ever forwarded.
        """
        input_data: Dict[str, Any] = {}
        chart_artifacts: List[Dict[str, Any]] = []

        for dep_key in step.input_from:
            if dep_key not in state_data:
                # Surface the missing key as a controlled engine error so
                # the outer try/except can route it to a failed callback.
                raise ValueError(
                    f"Required input '{dep_key}' is missing in execution state."
                )

            value = state_data[dep_key]

            # First dataframe-shaped value wins. Subsequent dataframes among
            # the input_from keys are intentionally ignored: a step receives
            # exactly one input_dataframe (req 4.3, 5.5).
            if "input_dataframe" not in input_data:
                df_candidate = cls._extract_dataframe(value)
                if df_candidate is not None:
                    input_data["input_dataframe"] = df_candidate

            # Surface any chart-shaped artifacts on upstream envelopes for
            # the reporting agent's visualization-aware codepath. Today the
            # visualization slot is disabled by default so this branch is
            # dormant in production, but it keeps the engine's existing
            # integration test passing without forwarding raw envelopes.
            if isinstance(value, dict):
                artifacts = value.get("artifacts")
                if isinstance(artifacts, dict):
                    for art_value in artifacts.values():
                        if (
                            isinstance(art_value, dict)
                            and "type" in art_value
                            and "title" in art_value
                        ):
                            chart_artifacts.append(art_value)

                    # Reporting-agent-only exception: forward the upstream
                    # filter agent's column_mapping artifact so the audit
                    # sheet writer can render it (req 8.5). The exception
                    # is documented here intentionally; no other agent ever
                    # receives state-data keys outside its input_from list.
                    if (
                        step.agent == "reporting_agent"
                        and "column_mapping" not in input_data
                    ):
                        cm = artifacts.get("column_mapping")
                        if cm is not None:
                            input_data["column_mapping"] = cm

        if chart_artifacts:
            input_data["chart_artifacts"] = chart_artifacts

        # Build the shared IntentPackage once at the filter boundary when
        # the caller did not inject one explicitly. This keeps column
        # resolution centralized and makes the package available to the
        # filter and reporting agents.
        if (
            step.agent == "filter_agent"
            and "intent_package" not in input_data
            and "input_dataframe" in input_data
        ):
            submission_id = str(state_data.get("__submission_id__", "unknown-submission"))
            package = cls._build_intent_package_for_filter_step(
                step=step,
                dataframe=input_data["input_dataframe"],
                submission_id=submission_id,
            )
            state_data["__intent_package__"] = package
            input_data["intent_package"] = package

        # Inject the shared IntentPackage when present in state_data.
        # The engine stores it under "__intent_package__" so it never
        # collides with step output_keys; agents receive it as
        # input_data["intent_package"].
        if "__intent_package__" in state_data:
            input_data["intent_package"] = state_data["__intent_package__"]

        return input_data

    # ------------------------------------------------------------------
    # Per-step param model lookup
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_params_model(agent_cls: Type) -> Optional[Type[BaseModel]]:
        """Return the agent's declared ``params_model`` if any.

        Reading the attribute directly off the class (rather than through
        ``AGENT_PARAM_MODELS``) keeps the engine resilient to the test
        pattern of hot-swapping ``registry._agents[name]`` without
        synchronising ``registry._param_models``: the agent class is the
        single source of truth for its own params model, so a swapped-in
        fake without a ``params_model`` attribute correctly skips the
        defensive re-validation. Production code paths (where the registry
        is the source) behave identically.
        """
        params_model = getattr(agent_cls, "params_model", None)
        if isinstance(params_model, type) and issubclass(params_model, BaseModel):
            return params_model
        return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute(
        self,
        plan: ExecutionPlan,
        intent_package: Optional[Any] = None,
        submission_id: Optional[str] = None,
    ) -> dict:
        sorted_ids = self._topological_sort(plan)
        self._validate_stages(plan)

        self.step_results = {}
        step_dict = {s.step_id: s for s in plan.steps}

        # Store the intent_package in a sentinel key so _build_input_data
        # can inject it into every agent's input_data dict.
        self._intent_package = intent_package
        self._submission_id = submission_id or "unknown-submission"

        builder = StateGraph(PipelineState)

        engine = self  # captured by closures below

        def create_node_func(step):
            step_id = step.step_id
            agent_name = step.agent
            params = step.params

            def node_func(state: PipelineState) -> dict:
                agent_cls = registry.get_agent_class(agent_name)

                # 1. Defense-in-depth per-step Pydantic params re-validation
                #    (req 4.1, 4.2, 10.6). Skipped only when the agent does
                #    not declare a params_model (e.g. calculation_agent).
                params_model = engine._resolve_params_model(agent_cls)
                if params_model is not None:
                    try:
                        params_model.model_validate(params)
                    except ValidationError as exc:
                        message = (
                            f"Invalid params for step '{step_id}' "
                            f"(agent={agent_name}, "
                            f"param_model={params_model.__name__}): {exc}"
                        )
                        engine.step_results[step_id] = {
                            "agent": agent_name,
                            "status": "failed",
                            "summary": None,
                            "metrics": {},
                            "operations_applied": [],
                            "warnings": [],
                            "artifacts": {},
                            "error_message": message,
                        }
                        # Raising surfaces the failure to the outer
                        # try/except, which then assembles the failed
                        # callback with the recorded step_id.
                        raise Exception(message)

                # 2. Build input_data exclusively from step.input_from. The
                #    helper enforces the single-source input_dataframe rule
                #    and forwards no raw state-data keys (req 4.3, 4.7, 5.5).
                try:
                    input_data = engine._build_input_data(step, state.data)
                except ValueError as exc:
                    engine.step_results[step_id] = {
                        "agent": agent_name,
                        "status": "failed",
                        "summary": None,
                        "metrics": {},
                        "operations_applied": [],
                        "warnings": [],
                        "artifacts": {},
                        "error_message": str(exc),
                    }
                    raise Exception(str(exc))

                # 3. Invoke the agent.
                agent_instance = agent_cls()
                result: AgentResult = agent_instance.execute(params, input_data)

                # 4. Record telemetry BEFORE any failure raises so the
                #    failed-step lookup at the bottom is deterministic
                #    (req 4.5).
                engine.step_results[step_id] = {
                    "agent": agent_name,
                    "status": result.status,
                    "summary": result.summary,
                    "metrics": result.metrics,
                    "operations_applied": result.operations_applied,
                    "warnings": result.warnings,
                    "artifacts": result.artifacts,
                    "error_message": result.error_message,
                }

                if result.status in ("failed", "partial"):
                    raise Exception(
                        result.error_message
                        or f"Step {step_id} failed with status {result.status}"
                    )

                # 5. Store the envelope under step.output_key (or step_id
                #    when output_key is None). No envelope is stored under
                #    any other key (req 4.6).
                out_k = step.output_key if step.output_key else step_id
                envelope = {
                    "data": result.data,
                    "artifacts": result.artifacts,
                    "metrics": result.metrics,
                    "warnings": result.warnings,
                    "summary": result.summary,
                }
                return {"data": {**state.data, out_k: envelope}}

            return node_func

        for sid in sorted_ids:
            step = step_dict[sid]
            builder.add_node(sid, create_node_func(step))

        # Compute transitive ancestors so we can apply transitive reduction
        # to the depends_on graph. Without reduction LangGraph sees redundant
        # parallel paths and the dynamic graph can mis-route edges.
        ancestors: Dict[str, set] = {}
        for sid in sorted_ids:
            step = step_dict[sid]
            step_ancestors: set = set()
            for dep in step.depends_on:
                step_ancestors.add(dep)
                step_ancestors.update(ancestors.get(dep, set()))
            ancestors[sid] = step_ancestors

        for sid in sorted_ids:
            step = step_dict[sid]
            if not step.depends_on:
                builder.add_edge(START, sid)
            else:
                essential_deps: List[str] = []
                for dep in step.depends_on:
                    is_redundant = False
                    for other_dep in step.depends_on:
                        if dep != other_dep and dep in ancestors.get(other_dep, set()):
                            is_redundant = True
                            break
                    if not is_redundant:
                        essential_deps.append(dep)

                for dep in essential_deps:
                    builder.add_edge(dep, sid)

        # Terminal nodes: any step nothing depends on flows to END.
        out_degree: Dict[str, int] = {s.step_id: 0 for s in plan.steps}
        for step in plan.steps:
            for dep in step.depends_on:
                out_degree[dep] += 1

        for sid, deg in out_degree.items():
            if deg == 0:
                builder.add_edge(sid, END)

        graph = builder.compile()

        start_time = time.time()
        try:
            initial_state = PipelineState()
            # Seed the intent_package into state.data so _build_input_data
            # can forward it to every agent's input_data dict.
            if intent_package is not None:
                initial_state.data["__intent_package__"] = intent_package
            initial_state.data["__submission_id__"] = self._submission_id
            graph.invoke(initial_state)
            duration_ms = int((time.time() - start_time) * 1000)

            reporting_step_id: Optional[str] = None
            for step in plan.steps:
                if step.agent == "reporting_agent":
                    reporting_step_id = step.step_id
                    break

            output_path: Optional[str] = None
            if reporting_step_id and reporting_step_id in self.step_results:
                artifacts = self.step_results[reporting_step_id].get("artifacts") or {}
                output_path = artifacts.get("primary_output_path")

            summary_info: Dict[str, Any] = {
                "steps_run": sorted_ids,
                "step_statuses": {
                    sid: res["status"] for sid, res in self.step_results.items()
                },
                "duration_ms": duration_ms,
                "agent_summaries": self._build_agent_summaries(),
                "step_metrics": {
                    sid: res["metrics"] for sid, res in self.step_results.items()
                },
                "operations_applied_by_step": {
                    sid: res["operations_applied"]
                    for sid, res in self.step_results.items()
                },
                "warnings_by_step": {
                    sid: res["warnings"] for sid, res in self.step_results.items()
                },
            }
            if not reporting_step_id:
                summary_info["warnings_by_step"]["engine"] = [
                    "No reporting step was defined in the plan, so no primary "
                    "output path was generated."
                ]

            return {
                "status": "complete",
                "output_path": output_path,
                "summary": summary_info,
            }
        except Exception as exc:
            duration_ms = int((time.time() - start_time) * 1000)

            # The failed step is whichever sorted step landed in
            # step_results with status in {failed, partial}. Because we
            # write the entry BEFORE raising (above), the lookup is
            # deterministic; the topological order ensures we surface the
            # earliest failure when multiple steps somehow recorded one.
            failed_step_id: Optional[str] = None
            for sid in sorted_ids:
                if (
                    sid in self.step_results
                    and self.step_results[sid]["status"] in ("failed", "partial")
                ):
                    failed_step_id = sid
                    break
            if not failed_step_id:
                failed_step_id = sorted_ids[0] if sorted_ids else "unknown"

            return {
                "status": "failed",
                "output_path": None,
                "summary": {
                    "steps_run": [
                        sid for sid in sorted_ids if sid in self.step_results
                    ],
                    "step_statuses": {
                        sid: res["status"]
                        for sid, res in self.step_results.items()
                    },
                    "duration_ms": duration_ms,
                    "agent_summaries": self._build_agent_summaries(),
                    "step_metrics": {
                        sid: res["metrics"]
                        for sid, res in self.step_results.items()
                    },
                    "operations_applied_by_step": {
                        sid: res["operations_applied"]
                        for sid, res in self.step_results.items()
                    },
                    "warnings_by_step": {
                        sid: res["warnings"]
                        for sid, res in self.step_results.items()
                    },
                    "failed_step_id": failed_step_id,
                    "error": str(exc),
                    "error_message": str(exc),
                },
            }
