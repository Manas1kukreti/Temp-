"""Deterministic compiler that turns validated intent into a fixed-shape
``ExecutionPlan``.

This compiler is the single source of truth for the canonical pipeline shape
described by the agent-pipeline-hardening spec (Component 6). It enforces:

* ``ingestion_agent`` is always the first step and ``reporting_agent`` is
  always the last step.
* A ``cleaning_agent`` step running in ``mode="clean"`` is emitted only when
  the intent requested cleaning AND supplied a ``cleaning_plan``. Its output
  is published under the canonical key ``df_cleaned``.
* When filtering is requested without cleaning, a non-destructive
  ``filter_prep`` step is inserted (realized as a ``cleaning_agent``
  invocation with ``mode="filter_prep"`` and the seven safe normalization
  operations). Its output is published under ``df_filter_prepared`` and the
  subsequent ``filter_agent`` step reads from that key, never from the raw
  ``df_ingested`` output.
* A ``visualization_agent`` step is only emitted when visualization is
  enabled at the process level. Otherwise the compiler raises
  :class:`VisualizationDisabledError`.
* Every emitted ``output_key`` is drawn from the canonical set
  ``{df_ingested, df_cleaned, df_filter_prepared, df_filtered,
  df_visualized, report_output}``.

Requirements satisfied: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10,
2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 9.2.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from finflow_agent.agents.visualization_agent import VISUALIZATION_DISABLED_MESSAGE
from finflow_agent.contract_registry import (
    validate_operator,
    validate_action_kind,
    resolve_semantic_operator,
    resolve_semantic_operation_type,
    InvalidOperatorError,
    InvalidActionKindError,
    UnmappedSemanticTypeError,
)
from finflow_agent.operations.schemas import (
    CalculationOperation,
    CalculationOperationPlan,
    CleaningOperationPlan,
    DropDuplicatesOperation,
    DropNullsOperation,
    FilterCondition,
    FilterOperationPlan,
    NormalizeColumnNamesOperation,
    RemoveEmptyRowsOperation,
    TrimWhitespaceOperation,
)
from finflow_agent.planning.canonical_intent import (
    CANONICAL_INTENT_SCHEMA_VERSION,
    CanonicalIntent,
    CalculateIntent,
    CleanIntent,
    DropColumnsIntent,
    FilterRowsIntent,
    LimitRowsIntent,
    ProjectColumnsIntent,
    ReportIntent,
    SortRowsIntent,
    VisualizeIntent,
    UnresolvedColumnReference,
)
from finflow_agent.planning.intent_schema import PlanIntent
from finflow_agent.state import ExecutionPlan, PlanStep
from finflow_agent.tools.config import get_enable_visualization

# ---------------------------------------------------------------------------
# Public types and constants
# ---------------------------------------------------------------------------


class SemanticCompilationError(Exception):
    """Raised when the compiler detects a contract violation during compilation.

    This error wraps contract-registry validation failures (invalid operators
    or invalid action kinds) so downstream consumers of the compiler have a
    single exception type to catch for all compilation failures. The message
    includes the invalid value and valid options for fast debugging.
    """


class VisualizationDisabledError(Exception):
    """Raised when a ``PlanIntent`` requests visualization while the
    visualization agent is disabled at the process level.

    The orchestrator catches this exception and converts it into a quarantine
    result (Requirement 9.3 / 11.6); it MUST NOT leak past the orchestrator.
    The canonical message is built from
    :data:`VISUALIZATION_DISABLED_MESSAGE` so the validator, compiler, and
    disabled agent placeholder all surface the same wording.
    """


# Exact message the compiler raises. Composed from the shared constant so the
# wording stays in lockstep with the validator and the visualization agent
# scaffold.
VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE: str = (
    f"Visualization was requested, but {VISUALIZATION_DISABLED_MESSAGE}"
)

# Canonical ``output_key`` set that the compiler is permitted to emit. Listed
# explicitly so the contract is unambiguous and the assertion in the
# post-conditions can use a frozen set.
CANONICAL_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "df_ingested",
        "df_cleaned",
        "df_filter_prepared",
        "df_filtered",
        "df_calculated",
        "df_visualized",
        "report_output",
    }
)

# The seven safe normalization operations the compiler emits for a
# ``filter_prep`` step. The order matches the spec wording so audit logs and
# downstream tests can rely on it.
SAFE_FILTER_PREP_OPERATIONS: List[str] = [
    "trim_whitespace",
    "normalize_column_names",
    "normalize_empty_strings",
    "safe_numeric_conversion",
    "safe_currency_conversion",
    "safe_date_detection",
    "categorical_value_normalization",
]


# ---------------------------------------------------------------------------
# Contract validation wrappers
# ---------------------------------------------------------------------------


def _validated_operator(operator: str) -> str:
    """Validate operator against contract registry, raise SemanticCompilationError on failure."""
    try:
        return validate_operator(operator).value
    except InvalidOperatorError as e:
        raise SemanticCompilationError(str(e)) from e


def _validated_action_kind(kind: str) -> str:
    """Validate action kind against contract registry, raise SemanticCompilationError on failure."""
    try:
        return validate_action_kind(kind).value
    except InvalidActionKindError as e:
        raise SemanticCompilationError(str(e)) from e


def _resolved_semantic_operator(relation_operator: str) -> str:
    """Resolve a semantic relation operator to its canonical operator via the contract registry.

    Wraps UnmappedSemanticTypeError as SemanticCompilationError so callers
    see a single exception type for all compilation failures.
    """
    try:
        return resolve_semantic_operator(relation_operator).value
    except UnmappedSemanticTypeError as e:
        raise SemanticCompilationError(str(e)) from e


def _resolved_semantic_operation_type(operation_type: str) -> str:
    """Resolve a semantic operation type to its canonical action kind via the contract registry.

    Wraps UnmappedSemanticTypeError as SemanticCompilationError so callers
    see a single exception type for all compilation failures.
    """
    try:
        return resolve_semantic_operation_type(operation_type).value
    except UnmappedSemanticTypeError as e:
        raise SemanticCompilationError(str(e)) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_plan_for_flag(intent: PlanIntent) -> None:
    """Raise ``ValueError`` when any ``needs_X`` flag lacks its matching plan.

    Requirement 2.13 requires the compiler to refuse intents that toggle a
    capability flag without supplying the matching plan, for every
    ``X ∈ {cleaning, filtering, calculation, visualization}``. The error
    message names the missing field so the orchestrator's quarantine reason
    is precise.
    """
    missing: List[str] = []
    if intent.needs_cleaning and intent.cleaning_plan is None:
        missing.append("cleaning_plan")
    if intent.needs_filtering and intent.filter_plan is None:
        missing.append("filter_plan")
    if intent.needs_calculation and intent.calculation_plan is None:
        missing.append("calculation_plan")
    if intent.needs_visualization and intent.visualization_plan is None:
        missing.append("visualization_plan")

    if missing:
        # Surface the first missing field in the message; all four checks are
        # deterministic so the order is stable across runs. The wording
        # ``needs_X is true but X_plan is missing`` names both the offending
        # flag and the missing field so the orchestrator's quarantine reason
        # is fully explanatory.
        field = missing[0]
        flag = "needs_" + field.replace("_plan", "")
        raise ValueError(
            f"{flag} is true but {field} is missing"
        )


def build_reporting_params(
    intent: PlanIntent,
    output_dir: str,
    file_prefix: str,
) -> Dict[str, Any]:
    """Build the ``params`` dict for the trailing ``reporting_agent`` step.

    Kept as a small helper so future enrichments (e.g. propagating reporting
    hints, audit-log markers) have a single attachment point that does not
    require touching the main compile function. The shape matches
    ``ReportingAgentParams`` (Requirement 10.4): a nested ``plan`` produced
    by ``ReportingOperationPlan`` plus the writer's optional output
    locators.
    """
    return {
        "plan": {
            "output_format": intent.output_format,
            "title": intent.reporting_title,
            "sheet_name": intent.sheet_name,
        },
        "output_dir": output_dir,
        "file_prefix": file_prefix,
    }


def _assert_canonical_output_keys(steps: List[PlanStep]) -> None:
    """Defensive post-condition: every emitted ``output_key`` is canonical.

    Requirement 2.14 forbids the compiler from publishing dataframes under
    any key outside ``CANONICAL_OUTPUT_KEYS``. The validator (sibling task
    8.1) re-checks this independently, but a compile-time assertion keeps
    the contract local to this function and surfaces regressions quickly.
    """
    for step in steps:
        if step.output_key is not None and step.output_key not in CANONICAL_OUTPUT_KEYS:
            raise ValueError(
                f"Compiler emitted non-canonical output_key {step.output_key!r} "
                f"on step {step.step_id!r}; allowed keys: "
                f"{sorted(CANONICAL_OUTPUT_KEYS)}"
            )


def _assert_filter_input_from(steps: List[PlanStep]) -> None:
    """Defensive post-condition: every ``filter_agent`` step reads from
    ``df_cleaned`` or ``df_filter_prepared`` and never from ``df_ingested``.

    Requirement 2.10 forbids the compiler from routing the raw ingestion
    output into the filter agent. The validator re-checks this rule, but a
    compile-time assertion makes accidental regressions in this module fail
    loudly during compilation rather than at validation time.
    """
    for step in steps:
        if step.agent != "filter_agent":
            continue
        if step.input_from == ["df_ingested"]:
            raise ValueError(
                f"Compiler routed raw df_ingested into filter step "
                f"{step.step_id!r}; this is forbidden by Requirement 2.10."
            )
        if step.input_from not in (["df_cleaned"], ["df_filter_prepared"]):
            raise ValueError(
                f"filter_agent step {step.step_id!r} has input_from="
                f"{step.input_from!r}; expected ['df_cleaned'] or "
                f"['df_filter_prepared']."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_intent_to_plan(
    intent: PlanIntent,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    file_prefix: str,
) -> ExecutionPlan:
    """Deterministically translate a validated ``PlanIntent`` into an
    ``ExecutionPlan`` whose shape obeys every clause of Requirement 2.

    The function is pure: identical inputs produce identical
    ``ExecutionPlan`` objects (modulo Python's dict ordering, which Pydantic
    preserves for ``model_dump`` here). All branches consult
    :func:`get_enable_visualization` rather than reading the environment
    directly so tests can monkeypatch the flag deterministically.

    See :mod:`finflow_agent.planning.compiler` module docstring for the full
    contract.
    """
    # ------------------------------------------------------------------
    # 0. Reject intents whose flags are inconsistent with their plans.
    #    Done first so the user sees the most actionable error when several
    #    issues are present (Requirement 2.13).
    # ------------------------------------------------------------------
    _require_plan_for_flag(intent)

    # PDF is not a supported output_format. PlanIntent's Literal type already
    # forbids it at validation time, but the orchestrator can mutate the
    # field after the fact (e.g. to surface a clearer error to the LLM), so
    # we keep an explicit guard here.
    if intent.output_format == "pdf":
        raise ValueError("PDF output format is not allowed.")

    steps: List[PlanStep] = []
    last_df_key: str = "df_ingested"

    # ------------------------------------------------------------------
    # 1. Ingestion is always the first step (Requirement 2.1).
    # ------------------------------------------------------------------
    steps.append(
        PlanStep(
            step_id="ingest",
            agent="ingestion_agent",
            params={
                "resolved_file_path": resolved_file_path,
                "file_type": file_type,
            },
            depends_on=[],
            input_from=[],
            output_key="df_ingested",
        )
    )

    # ------------------------------------------------------------------
    # 2. Filter pipeline branch.
    #
    #    Three mutually exclusive sub-cases, in priority order:
    #      (a) needs_filtering AND needs_cleaning  -> ingest, clean, filter
    #      (b) needs_filtering AND NOT needs_cleaning -> ingest, filter_prep, filter
    #      (c) NOT needs_filtering AND needs_cleaning -> ingest, clean
    #    The two filtering sub-cases share the trailing ``filter_agent``
    #    step but differ in which preparation step feeds it.
    # ------------------------------------------------------------------
    if intent.needs_filtering and intent.filter_plan is not None:
        if intent.needs_cleaning and intent.cleaning_plan is not None:
            # 2a. Full cleaning requested; emit a destructive cleaning_agent
            #     step in ``mode="clean"`` (Requirement 2.2).
            steps.append(
                PlanStep(
                    step_id="clean",
                    agent="cleaning_agent",
                    params={
                        "plan": intent.cleaning_plan.model_dump(),
                        "mode": "clean",
                    },
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_cleaned",
                )
            )
            filter_input = "df_cleaned"
            filter_depends_on = "clean"
            last_df_key = "df_cleaned"
        else:
            # 2b. Filtering without cleaning; insert a non-destructive
            #     filter_prep step (Requirements 2.6, 2.7). The step is a
            #     cleaning_agent invocation with ``mode="filter_prep"`` and
            #     ONLY the seven safe operations enumerated in
            #     ``SAFE_FILTER_PREP_OPERATIONS``. ``params`` deliberately
            #     omits ``plan`` per the task contract; the cleaning_agent
            #     dispatch (sibling task 5.1) handles the ``filter_prep``
            #     mode using the operations list.
            steps.append(
                PlanStep(
                    step_id="filter_prep",
                    agent="cleaning_agent",
                    params={
                        "mode": "filter_prep",
                        "operations": list(SAFE_FILTER_PREP_OPERATIONS),
                    },
                    depends_on=["ingest"],
                    input_from=["df_ingested"],
                    output_key="df_filter_prepared",
                )
            )
            filter_input = "df_filter_prepared"
            filter_depends_on = "filter_prep"
            last_df_key = "df_filter_prepared"

        # The filter step itself. Requirement 2.10: ``input_from`` must be
        # exactly one of ``["df_cleaned"]`` or ``["df_filter_prepared"]``,
        # never ``["df_ingested"]``. The post-condition assertion below
        # re-checks this.
        steps.append(
            PlanStep(
                step_id="filter",
                agent="filter_agent",
                params={"plan": intent.filter_plan.model_dump()},
                depends_on=[filter_depends_on],
                input_from=[filter_input],
                output_key="df_filtered",
            )
        )
        last_df_key = "df_filtered"

    elif intent.needs_cleaning and intent.cleaning_plan is not None:
        # 2c. Clean-only branch (no filtering requested).
        steps.append(
            PlanStep(
                step_id="clean",
                agent="cleaning_agent",
                params={
                    "plan": intent.cleaning_plan.model_dump(),
                    "mode": "clean",
                },
                depends_on=["ingest"],
                input_from=["df_ingested"],
                output_key="df_cleaned",
            )
        )
        last_df_key = "df_cleaned"

    # ------------------------------------------------------------------
    # 2.5. Calculation step (after filter/clean, before visualization).
    # ------------------------------------------------------------------
    if intent.needs_calculation and intent.calculation_plan is not None:
        steps.append(
            PlanStep(
                step_id="calculate",
                agent="calculation_agent",
                params={
                    "operations": [
                        op.model_dump() if hasattr(op, "model_dump") else op
                        for op in intent.calculation_plan.operations
                    ],
                },
                depends_on=[steps[-1].step_id],
                input_from=[last_df_key],
                output_key="df_calculated",
            )
        )
        last_df_key = "df_calculated"

    # ------------------------------------------------------------------
    # 3. Visualization (scaffolded; gated by ENABLE_VISUALIZATION).
    #
    #    Only entered when the intent both requested visualization and
    #    supplied a plan; the missing-plan branch was already raised in
    #    step 0. The flag is read via ``get_enable_visualization`` so test
    #    fixtures can monkeypatch it deterministically.
    # ------------------------------------------------------------------
    if intent.needs_visualization and intent.visualization_plan is not None:
        if not get_enable_visualization():
            raise VisualizationDisabledError(
                VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE
            )
        steps.append(
            PlanStep(
                step_id="visualize",
                agent="visualization_agent",
                params={"plan": intent.visualization_plan.model_dump()},
                depends_on=[steps[-1].step_id],
                input_from=[last_df_key],
                output_key="df_visualized",
            )
        )
        last_df_key = "df_visualized"

    # ------------------------------------------------------------------
    # 4. Reporting is always the last step (Requirement 2.1).
    # ------------------------------------------------------------------
    steps.append(
        PlanStep(
            step_id="report",
            agent="reporting_agent",
            params=build_reporting_params(intent, output_dir, file_prefix),
            depends_on=[steps[-1].step_id],
            input_from=[last_df_key],
            output_key="report_output",
        )
    )

    plan = ExecutionPlan(steps=steps)

    # ------------------------------------------------------------------
    # 5. Sanity post-conditions. The validator re-checks all of these but
    #    asserting them here pins the contract local to the compiler and
    #    surfaces regressions in this module immediately.
    # ------------------------------------------------------------------
    assert plan.steps[0].agent == "ingestion_agent", (
        "Compiler post-condition violated: first step must be ingestion_agent"
    )
    assert plan.steps[-1].agent == "reporting_agent", (
        "Compiler post-condition violated: last step must be reporting_agent"
    )
    _assert_canonical_output_keys(plan.steps)
    _assert_filter_input_from(plan.steps)

    return plan


def compile_canonical_intent(
    intent: CanonicalIntent,
    *,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    artifact_prefix: str,
) -> ExecutionPlan:
    """Compile a validated ``CanonicalIntent`` into an ``ExecutionPlan``.

    This is the production entrypoint for worker-side execution planning.
    It does not accept raw prompts or free-form dicts.
    """
    _validate_canonical_intent(intent)
    plan_intent = _canonical_intent_to_plan_intent(
        intent,
        resolved_file_path=resolved_file_path,
        file_type=file_type,
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
    )
    return compile_intent_to_plan(
        intent=plan_intent,
        resolved_file_path=resolved_file_path,
        file_type=file_type,
        output_dir=output_dir,
        file_prefix=artifact_prefix,
    )


def _validate_canonical_intent(intent: CanonicalIntent) -> None:
    if intent.schema_version != CANONICAL_INTENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported canonical intent schema_version {intent.schema_version!r}; "
            f"expected {CANONICAL_INTENT_SCHEMA_VERSION!r}."
        )
    if intent.output_format not in {"xlsx", "csv", "json", "txt"}:
        raise ValueError(f"Unsupported output_format {intent.output_format!r}.")
    if intent.resolution_status not in {"resolved", "repaired"}:
        raise ValueError(
            f"Canonical intent resolution_status {intent.resolution_status!r} is not executable."
        )
    if not intent.actions:
        raise ValueError("Canonical intent contains no executable actions.")


def _canonical_intent_to_plan_intent(
    intent: CanonicalIntent,
    *,
    resolved_file_path: str,
    file_type: str,
    output_dir: str,
    artifact_prefix: str,
) -> PlanIntent:
    source_columns = _source_columns_from_intent(intent, resolved_file_path)

    cleaning_plan: CleaningOperationPlan | None = None
    filter_plan: FilterOperationPlan | None = None
    calculation_plan: CalculationOperationPlan | None = None
    needs_cleaning = False
    needs_filtering = False
    needs_calculation = False
    needs_visualization = False

    selected_columns: list[str] = []
    filter_conditions: list[FilterCondition] = []
    filter_logic = "and"
    filter_limit: int | None = None
    drop_columns_seen = False
    calculation_operations: list = []

    for action in intent.actions:
        # Resolve the semantic operation type to a canonical action kind via
        # the contract registry.  This ensures the compiler uses the single
        # source-of-truth mapping (Requirements 5.2, 6.1, 7.4).
        _resolved_semantic_operation_type(action.kind)

        if isinstance(action, CleanIntent):
            cleaning_plan = _build_cleaning_plan(action)
            needs_cleaning = True
            continue

        if isinstance(action, ProjectColumnsIntent):
            selected_columns = _resolve_requested_columns(action.requested_fields)
            needs_filtering = True
            continue

        if isinstance(action, DropColumnsIntent):
            drop_columns_seen = True
            if not source_columns:
                raise ValueError("drop_columns canonical intent requires source columns in the dataframe profile.")
            dropped = _resolve_requested_columns(action.requested_fields)
            selected_columns = [column for column in source_columns if column not in set(dropped)]
            needs_filtering = True
            continue

        if isinstance(action, FilterRowsIntent):
            filter_logic = action.logic
            needs_filtering = True
            filter_conditions.extend(_canonical_filter_conditions(action.conditions, action.mode))
            continue

        if isinstance(action, LimitRowsIntent):
            filter_limit = max(0, int(action.limit))
            needs_filtering = True
            continue

        if isinstance(action, SortRowsIntent):
            raise ValueError("Canonical sort_rows intent is not supported by the current compiler.")
        if isinstance(action, CalculateIntent):
            ops = _build_calculation_operations(action)
            if ops:
                needs_calculation = True
                calculation_operations = ops
            continue
        if isinstance(action, VisualizeIntent):
            needs_visualization = True
            raise ValueError("Canonical visualize intent is not supported by the current compiler.")
        if isinstance(action, ReportIntent):
            continue

        raise ValueError(f"Unsupported canonical intent action kind: {getattr(action, 'kind', type(action).__name__)}")

    if drop_columns_seen and not source_columns:
        raise ValueError("drop_columns canonical intent cannot be compiled without source columns.")

    if selected_columns or filter_conditions or filter_limit is not None:
        filter_plan = FilterOperationPlan(
            conditions=filter_conditions,
            logic=filter_logic,
            select_columns=selected_columns or None,
            limit=filter_limit,
        )
        needs_filtering = True

    if calculation_operations:
        calculation_plan = CalculationOperationPlan(
            operations=[
                CalculationOperation(**op) if isinstance(op, dict) else op
                for op in calculation_operations
            ]
        )

    return PlanIntent(
        needs_cleaning=needs_cleaning,
        needs_filtering=needs_filtering,
        needs_calculation=needs_calculation,
        needs_visualization=needs_visualization,
        output_format=intent.output_format,
        cleaning_plan=cleaning_plan,
        filter_plan=filter_plan,
        calculation_plan=calculation_plan,
        reporting_title=intent.decision or None,
        sheet_name=None,
    )


def _build_calculation_operations(action: CalculateIntent) -> list:
    """Build calculation operations from a CalculateIntent action."""
    operations = []
    for op in action.operations:
        if isinstance(op, dict):
            operations.append(op)
        elif isinstance(op, str):
            # Legacy format: just a string description — skip
            continue
        else:
            operations.append(op)
    return operations


def _source_columns_from_intent(intent: CanonicalIntent, resolved_file_path: str) -> list[str]:
    profile_columns = intent.dataframe_profile.get("source_columns")
    if not isinstance(profile_columns, list):
        # Fallback: some pipelines emit "columns" instead of "source_columns"
        profile_columns = intent.dataframe_profile.get("columns")
    if isinstance(profile_columns, list):
        columns = [str(column).strip() for column in profile_columns if str(column).strip()]
        if columns:
            return columns

    path = Path(str(resolved_file_path))
    if not path.exists():
        return []

    extension = path.suffix.lower()
    try:
        if extension in {".csv", ".tsv"}:
            frame = pd.read_csv(path, nrows=0, sep="\t" if extension == ".tsv" else ",")
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, nrows=0)
        else:
            return []
        return [str(column).strip() for column in frame.columns if str(column).strip()]
    except Exception:
        return []


def _resolve_requested_columns(fields: list[UnresolvedColumnReference]) -> list[str]:
    resolved: list[str] = []
    for field in fields:
        resolved.extend(_resolve_field_columns(field))
    return list(dict.fromkeys(resolved))


def _resolve_field_columns(field: UnresolvedColumnReference) -> list[str]:
    if field.resolved_columns:
        columns = [str(column).strip() for column in field.resolved_columns if str(column).strip()]
        if columns:
            return columns

    column = str(field.resolved_column or "").strip()
    if column:
        return [column]

    candidates = [str(column).strip() for column in field.candidate_columns if str(column).strip()]
    detail = f"selection_mode={field.selection_mode!r}"
    if candidates:
        detail += f", candidates={candidates!r}"
    raise ValueError(f"Unresolved canonical column reference: {field.raw_reference!r} ({detail})")


def _build_cleaning_plan(action: CleanIntent) -> CleaningOperationPlan:
    if not action.operations:
        return CleaningOperationPlan(
            operations=[
                TrimWhitespaceOperation(columns="__all_string_columns__"),
                NormalizeColumnNamesOperation(style="snake_case"),
                DropDuplicatesOperation(subset=None, keep="first"),
            ]
        )

    operations: list[Any] = []
    for operation in action.operations:
        name = operation.name.strip()
        if name == "trim_whitespace":
            operations.append(TrimWhitespaceOperation(columns="__all_string_columns__"))
        elif name == "normalize_column_names":
            operations.append(NormalizeColumnNamesOperation(style="snake_case"))
        elif name == "drop_duplicates":
            operations.append(DropDuplicatesOperation(subset=None, keep="first"))
        elif name == "drop_nulls":
            columns = operation.parameters.get("columns") if isinstance(operation.parameters, dict) else None
            how = str(operation.parameters.get("how", "any")).strip().lower() if isinstance(operation.parameters, dict) else "any"
            if columns is not None and not isinstance(columns, list):
                raise ValueError("drop_nulls cleaning operation requires columns to be null or a list of column names.")
            operations.append(DropNullsOperation(columns=columns, how=how if how in {"any", "all"} else "any"))
        elif name == "remove_empty_rows":
            operations.append(RemoveEmptyRowsOperation())
        else:
            raise ValueError(f"Unsupported canonical cleaning operation: {name!r}")
    return CleaningOperationPlan(operations=operations)


def _canonical_filter_conditions(
    conditions: list[FilterCondition],
    mode: str,
) -> list[FilterCondition]:
    if mode == "keep":
        result = []
        for condition in conditions:
            # Skip conditions with unresolved generic field references
            if not _can_resolve_field(condition.field):
                continue
            result.append(
                FilterCondition(
                    column=_resolve_single_grounded_column(condition.field),
                    operator=_resolved_semantic_operator(condition.operator),
                    value=condition.value,
                )
            )
        return result

    inverted: list[FilterCondition] = []
    for condition in conditions:
        if not _can_resolve_field(condition.field):
            continue
        operator = _invert_operator(condition.operator)
        inverted.append(
            FilterCondition(
                column=_resolve_single_grounded_column(condition.field),
                operator=_resolved_semantic_operator(operator),
                value=condition.value,
            )
        )
    return inverted


def _can_resolve_field(field) -> bool:
    """Check if a field reference can be resolved to a column name."""
    if hasattr(field, "resolved_column") and field.resolved_column:
        return True
    if hasattr(field, "resolved_columns") and field.resolved_columns:
        return True
    # Generic/unresolved references cannot be compiled
    raw_ref = str(getattr(field, "raw_reference", "") or "").strip().lower()
    if raw_ref in {"generic", "", "unknown", "null", "none"}:
        return False
    # If resolution_method is generic_reference with no resolved column, skip
    method = str(getattr(field, "resolution_method", "") or "").strip().lower()
    if method == "generic_reference":
        return False
    return False


def _resolve_single_grounded_column(field: UnresolvedColumnReference) -> str:
    column = str(field.resolved_column or "").strip()
    if column:
        return column
    if field.resolved_columns:
        raise ValueError(
            f"Canonical filter field {field.raw_reference!r} resolved to multiple columns {field.resolved_columns!r}; "
            "filter conditions require a single grounded column."
        )
    raise ValueError(f"Unresolved canonical filter field: {field.raw_reference!r}")


def _invert_operator(operator: str) -> str:
    mapping = {
        "eq": "neq",
        "neq": "eq",
        "gt": "lte",
        "gte": "lt",
        "lt": "gte",
        "lte": "gt",
        "contains": "not_contains",
    }
    if operator not in mapping:
        raise ValueError(f"Unsupported operator for drop-mode canonical intent: {operator!r}")
    return mapping[operator]


__all__ = [
    "SemanticCompilationError",
    "VisualizationDisabledError",
    "VISUALIZATION_REQUESTED_BUT_DISABLED_MESSAGE",
    "CANONICAL_OUTPUT_KEYS",
    "SAFE_FILTER_PREP_OPERATIONS",
    "build_reporting_params",
    "compile_intent_to_plan",
    # --- New refactored pipeline exports (Semantic Grounding Refactor) ---
    "CompilerError",
    "ExecutionStep",
    "RefactoredExecutionPlan",
    "Compiler",
]


# ===========================================================================
# NEW: Refactored Compiler (Semantic Grounding Refactor)
#
# This section implements the deterministic Compiler contract from the
# semantic-grounding-refactor spec. It accepts only the new CanonicalIntent
# model (from models/canonical.py), validates all column references are
# resolved (type-level guarantee), and produces an ExecutionPlan.
#
# Requirements: 6.1, 6.4, 11.1, 11.2
# ===========================================================================

from enum import Enum
from typing import Literal as TypingLiteral
from uuid import uuid4 as _uuid4

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.canonical import (
    CanonicalIntent as RefactoredCanonicalIntent,
    ResolvedAction,
    ResolvedFilterAction,
    ResolvedProjectAction,
    ResolvedDropAction,
    ResolvedSortAction,
    ResolvedRenameAction,
)


class CompilerError(Exception):
    """Raised when the Compiler detects a contract violation.

    This covers:
    - Input that is not a CanonicalIntent
    - Column references that are not resolved to physical columns
    - Any attempt to perform grounding or semantic re-interpretation

    Requirements: 11.1, 11.2
    """

    pass


class ActionType(str, Enum):
    """Enumeration of execution step action types."""

    FILTER = "filter"
    PROJECT = "project"
    DROP = "drop"
    SORT = "sort"
    RENAME = "rename"


class ExecutionStep(BaseModel):
    """A single execution step in the execution plan.

    Each step represents one resolved action that the Executor will perform.
    All column references are physical (resolved) column names.

    Requirements: 11.1 - every column reference validated as physical.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    step_id: str = Field(default_factory=lambda: str(_uuid4()))
    action_type: ActionType
    resolved_columns: List[str] = Field(
        ...,
        description="Physical column names involved in this step",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (operators, values, directions, etc.)",
    )


class RefactoredExecutionPlan(BaseModel):
    """An execution plan produced by the refactored Compiler.

    Contains a list of ExecutionSteps plus intent metadata for traceability.
    The Executor walks these steps sequentially — no LLM calls, no grounding,
    no semantic interpretation.

    Requirements: 6.1, 6.4, 11.1
    """

    model_config = ConfigDict(strict=True, frozen=True)

    plan_id: str = Field(default_factory=lambda: str(_uuid4()))
    intent_id: str = Field(
        ..., description="ID of the CanonicalIntent that produced this plan"
    )
    source_draft_id: str = Field(
        ..., description="ID of the original SemanticIntentDraft"
    )
    source_draft_revision: int = Field(
        ..., description="Revision of the original draft"
    )
    schema_version: str = Field(
        default="1.0", description="Plan schema version"
    )
    steps: List[ExecutionStep] = Field(
        ..., description="Ordered list of execution steps"
    )


class Compiler:
    """Deterministic compiler: CanonicalIntent → ExecutionPlan.

    This compiler is the Decision_Owner for execution step selection.
    It accepts only CanonicalIntent objects (from models/canonical.py),
    which by type-level guarantee contain no unresolved active execution
    references. It validates all column references are resolved to
    physical columns and produces an ExecutionPlan.

    Contract:
    - Accepts ONLY CanonicalIntent (type-level guarantee, Req 11.2)
    - Validates every column reference resolved to physical column (Req 11.1)
    - Does NOT perform grounding (Req 6.4)
    - Does NOT make LLM calls (Req 6.4)
    - Does NOT re-interpret semantics (Req 6.4)
    - Produces a RefactoredExecutionPlan

    Requirements: 6.1, 6.4, 11.1, 11.2
    """

    def compile(self, intent: RefactoredCanonicalIntent) -> RefactoredExecutionPlan:
        """Compile a canonical intent into an execution plan.

        Preconditions (type-level guarantees):
        - Input is CanonicalIntent (always resolved)
        - Every column reference resolved to physical column

        Does NOT: perform grounding, make LLM calls, re-interpret semantics.

        Args:
            intent: A fully-resolved CanonicalIntent.

        Returns:
            A RefactoredExecutionPlan with ordered ExecutionSteps.

        Raises:
            CompilerError: If the input is not a CanonicalIntent or any
                column reference is not resolved.
        """
        # Type-level contract enforcement (Req 11.2)
        if not isinstance(intent, RefactoredCanonicalIntent):
            raise CompilerError(
                f"Compiler accepts only CanonicalIntent objects. "
                f"Received: {type(intent).__name__}"
            )

        # Validate resolution_status (redundant due to type literal, but
        # belt-and-suspenders for contract enforcement)
        if intent.resolution_status != "resolved":
            raise CompilerError(
                f"CanonicalIntent must have resolution_status='resolved'. "
                f"Got: {intent.resolution_status!r}"
            )

        # Validate actions are present
        if not intent.actions:
            raise CompilerError(
                "CanonicalIntent contains no actions to compile."
            )

        # Walk resolved actions and compile to execution steps
        steps: List[ExecutionStep] = []
        for action in intent.actions:
            step = self._compile_action(action)
            steps.append(step)

        return RefactoredExecutionPlan(
            intent_id=intent.intent_id,
            source_draft_id=intent.source_draft_id,
            source_draft_revision=intent.source_draft_revision,
            steps=steps,
        )

    def _compile_action(self, action: ResolvedAction) -> ExecutionStep:
        """Compile a single resolved action into an ExecutionStep.

        Validates that all column references within the action are resolved
        to physical column names (non-empty strings).

        Raises CompilerError if any column reference is invalid.
        """
        if isinstance(action, ResolvedFilterAction):
            return self._compile_filter(action)
        elif isinstance(action, ResolvedProjectAction):
            return self._compile_project(action)
        elif isinstance(action, ResolvedDropAction):
            return self._compile_drop(action)
        elif isinstance(action, ResolvedSortAction):
            return self._compile_sort(action)
        elif isinstance(action, ResolvedRenameAction):
            return self._compile_rename(action)
        else:
            raise CompilerError(
                f"Unknown resolved action type: {type(action).__name__}"
            )

    def _compile_filter(self, action: ResolvedFilterAction) -> ExecutionStep:
        """Compile a resolved filter action into an ExecutionStep."""
        columns: List[str] = []
        for predicate in action.predicates:
            col = predicate.get("column")
            if not col or not isinstance(col, str) or not col.strip():
                raise CompilerError(
                    f"Filter predicate has unresolved or empty column reference: "
                    f"{predicate!r}"
                )
            columns.append(str(col))

        self._validate_columns_resolved(columns, "filter")

        return ExecutionStep(
            action_type=ActionType.FILTER,
            resolved_columns=columns,
            params={"predicates": [dict(p) for p in action.predicates]},
        )

    def _compile_project(self, action: ResolvedProjectAction) -> ExecutionStep:
        """Compile a resolved project action into an ExecutionStep."""
        self._validate_columns_resolved(action.columns, "project")

        return ExecutionStep(
            action_type=ActionType.PROJECT,
            resolved_columns=list(action.columns),
            params={},
        )

    def _compile_drop(self, action: ResolvedDropAction) -> ExecutionStep:
        """Compile a resolved drop action into an ExecutionStep."""
        self._validate_columns_resolved(action.columns, "drop")

        return ExecutionStep(
            action_type=ActionType.DROP,
            resolved_columns=list(action.columns),
            params={},
        )

    def _compile_sort(self, action: ResolvedSortAction) -> ExecutionStep:
        """Compile a resolved sort action into an ExecutionStep."""
        self._validate_columns_resolved(action.keys, "sort")

        if len(action.keys) != len(action.directions):
            raise CompilerError(
                f"Sort action has mismatched keys ({len(action.keys)}) "
                f"and directions ({len(action.directions)})."
            )

        return ExecutionStep(
            action_type=ActionType.SORT,
            resolved_columns=list(action.keys),
            params={"directions": list(action.directions)},
        )

    def _compile_rename(self, action: ResolvedRenameAction) -> ExecutionStep:
        """Compile a resolved rename action into an ExecutionStep."""
        source_columns: List[str] = []
        rename_map: Dict[str, str] = {}

        for source_col, new_name in action.mappings:
            if not source_col or not source_col.strip():
                raise CompilerError(
                    f"Rename action has unresolved source column: "
                    f"'{source_col}' -> '{new_name}'"
                )
            if not new_name or not new_name.strip():
                raise CompilerError(
                    f"Rename action has empty target name for column: "
                    f"'{source_col}'"
                )
            source_columns.append(source_col)
            rename_map[source_col] = new_name

        self._validate_columns_resolved(source_columns, "rename")

        return ExecutionStep(
            action_type=ActionType.RENAME,
            resolved_columns=source_columns,
            params={"rename_map": rename_map},
        )

    @staticmethod
    def _validate_columns_resolved(
        columns: List[str], context: str
    ) -> None:
        """Validate that all column names are non-empty resolved physical columns.

        Raises CompilerError if any column is empty or whitespace-only.

        Requirements: 11.1 - every column reference validated as physical.
        """
        for col in columns:
            if not col or not col.strip():
                raise CompilerError(
                    f"Unresolved or empty column reference in {context} action: "
                    f"'{col}'. All columns must be resolved to physical names "
                    f"before compilation."
                )
