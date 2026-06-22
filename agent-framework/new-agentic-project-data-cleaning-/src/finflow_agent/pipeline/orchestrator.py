"""Pipeline orchestrator: wires all semantic pipeline stages together.

Connects all stages in order:
Extractor → Coverage Validator → Repair → Coordinator → Preflight → Schema →
Candidate Gen → Column Grounder + Predicate Grounder → Canonicalizer →
Compiler → Executor

Implements:
- Stage-resume on clarification (Req 9.3)
- Resolution policies for each uncertainty class (Req 10.1–10.6)
- Feature-flag-aware routing via LegacyRouter (Req 13.1, 13.2)
- Observability throughout (PipelineMetrics, PipelineTracingContext)

Requirements: 6.1, 6.4, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.grounding.candidate_generator import CandidateGenerator
from finflow_agent.grounding.column_grounder import ColumnGrounder
from finflow_agent.grounding.evidence import (
    ColumnGroundingResult,
    GroundingConfig,
    GroundingMethod,
    PredicateGroundingResult,
    ScoredCandidate,
)
from finflow_agent.grounding.llm_adapter import (
    LLMProviderError,
    LLMValidationError,
    SemanticResolver,
)
from finflow_agent.grounding.predicate_grounder import PredicateGrounder
from finflow_agent.grounding.preflight_loader import PreflightDataLoader
from finflow_agent.grounding.schema_service import SchemaService
from finflow_agent.grounding.semantic_extractor import (
    ExtractionError,
    SemanticExtractor,
)
from finflow_agent.models.canonical import CanonicalIntent, CanonicalizeError
from finflow_agent.models.draft import (
    FilterAction,
    ProjectAction,
    DropAction,
    RenameAction,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
    UnresolvedPredicate,
)
from finflow_agent.models.envelope import IntentEnvelope, PipelineStatus
from finflow_agent.pipeline.canonicalizer import Canonicalizer
from finflow_agent.pipeline.clarification_service import (
    ClarificationDraftPatcher,
    ClarificationResponse,
    StageResumeDirective,
)
from finflow_agent.pipeline.coordinator import IntentResolutionCoordinator
from finflow_agent.pipeline.coverage_validator import (
    CoverageValidationResult,
    CoverageValidator,
)
from finflow_agent.pipeline.feature_flags import FeatureFlags
from finflow_agent.pipeline.legacy_router import LegacyRouter, PipelineRoute
from finflow_agent.pipeline.observability import (
    PipelineMetrics,
    PipelineStage,
    PipelineTracingContext,
)
from finflow_agent.pipeline.patch_applicator import apply_patches
from finflow_agent.pipeline.resolution_handler import (
    mark_interpretation_failed,
    mark_needs_clarification,
    mark_resolved,
    set_resolution_origin,
)
from finflow_agent.pipeline.semantic_repair import (
    RepairAlreadyAttemptedError,
    SemanticRepair,
)
from finflow_agent.models.draft import ResolutionOrigin

logger = logging.getLogger(__name__)


def _import_compiler():
    """Lazy import to avoid circular import through planning → state → execution.

    The legacy compiler.py imports from finflow_agent.agents.visualization_agent
    which can trigger a circular import chain with finflow_agent.state and
    finflow_agent.execution. We handle this gracefully.
    """
    try:
        from finflow_agent.planning.compiler import (
            Compiler,
            CompilerError,
            RefactoredExecutionPlan,
        )
        return Compiler, CompilerError, RefactoredExecutionPlan
    except ImportError:
        # Circular import in legacy code — fall back to importing only what
        # we need from the module by loading execution.state first to break cycle
        import importlib
        import importlib.util
        import sys
        import os

        # src_dir is the directory containing the finflow_agent package
        src_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Pre-load execution.state to break circular dependency
        if "finflow_agent.execution.state" not in sys.modules:
            state_path = os.path.join(
                src_dir, "finflow_agent", "execution", "state.py"
            )
            spec = importlib.util.spec_from_file_location(
                "finflow_agent.execution.state",
                state_path,
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["finflow_agent.execution.state"] = mod
                spec.loader.exec_module(mod)

                # Also populate finflow_agent.state
                if "finflow_agent.state" not in sys.modules:
                    sys.modules["finflow_agent.state"] = mod

        # Now the circular import should be broken
        from finflow_agent.planning.compiler import (
            Compiler,
            CompilerError,
            RefactoredExecutionPlan,
        )
        return Compiler, CompilerError, RefactoredExecutionPlan


def _import_executor():
    """Lazy import to avoid circular import through execution → state."""
    try:
        from finflow_agent.execution.engine import (
            Executor,
            ExecutionResult,
            ExecutorIntentPackage,
            ColumnNotInPackageError,
            ContentHashMismatchError,
        )
        return Executor, ExecutionResult, ExecutorIntentPackage, ColumnNotInPackageError, ContentHashMismatchError
    except ImportError:
        # Same circular import workaround
        import importlib
        import importlib.util
        import sys
        import os

        # src_dir is the directory containing the finflow_agent package
        src_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Pre-load execution.state to break circular dependency
        if "finflow_agent.execution.state" not in sys.modules:
            state_path = os.path.join(
                src_dir, "finflow_agent", "execution", "state.py"
            )
            spec = importlib.util.spec_from_file_location(
                "finflow_agent.execution.state",
                state_path,
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["finflow_agent.execution.state"] = mod
                spec.loader.exec_module(mod)

                if "finflow_agent.state" not in sys.modules:
                    sys.modules["finflow_agent.state"] = mod

        from finflow_agent.execution.engine import (
            Executor,
            ExecutionResult,
            ExecutorIntentPackage,
            ColumnNotInPackageError,
            ContentHashMismatchError,
        )
        return Executor, ExecutionResult, ExecutorIntentPackage, ColumnNotInPackageError, ContentHashMismatchError


# ---------------------------------------------------------------------------
# Pipeline Result Model
# ---------------------------------------------------------------------------


class PipelineResultStatus(str, Enum):
    """Status of the pipeline execution result."""

    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ClarificationNeeded(BaseModel):
    """Details about what clarification is needed from the user."""

    model_config = ConfigDict(strict=True)

    reason: str
    candidates: list[str] = Field(default_factory=list)
    element_path: str = ""
    resume_directive: StageResumeDirective | None = None


class PipelineResult(BaseModel):
    """Result of the semantic pipeline execution.

    Contains either:
    - A resolved canonical_intent + execution plan (status=resolved)
    - A clarification_needed descriptor (status=needs_clarification)
    - An error description (status=failed or unsupported)
    """

    model_config = ConfigDict(strict=True)

    status: PipelineResultStatus
    canonical_intent: CanonicalIntent | None = None
    execution_plan: Any = None  # RefactoredExecutionPlan (lazy imported)
    execution_result: Any = None  # ExecutionResult (lazy imported)
    clarification_needed: ClarificationNeeded | None = None
    error: str | None = None
    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SemanticPipeline Orchestrator
# ---------------------------------------------------------------------------


class SemanticPipeline:
    """Full semantic pipeline orchestrator wiring all stages together.

    Pipeline stages in order:
    1. Extraction (SemanticExtractor)
    2. Coverage Validation (CoverageValidator)
    3. Bounded Repair (SemanticRepair)
    4. Resolution Coordination (IntentResolutionCoordinator)
    5. Preflight Data Load (PreflightDataLoader)
    6. Schema Inference (SchemaService)
    7. Candidate Generation (CandidateGenerator)
    8. Column Grounding + Predicate Grounding
    9. Canonicalization (Canonicalizer)
    10. Compilation (Compiler)
    11. Execution (Executor)

    Resolution policies (Req 10.1–10.6):
    - Formatting uncertainty → warning + proceed (10.1)
    - Semantic column uncertainty → Clarification (10.2)
    - Operation ambiguity → Clarification (10.3)
    - No match → unsupported or Clarification (10.4)
    - Contract violation → fail closed (10.5)
    - Low-confidence → clarify or fail (10.6)

    Requirements: 6.1, 6.4, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """

    def __init__(
        self,
        resolver: SemanticResolver,
        feature_flags: FeatureFlags,
    ) -> None:
        """Initialize the pipeline with all component instances.

        Args:
            resolver: LLM adapter for extraction, repair, and grounding.
            feature_flags: Configuration controlling new vs legacy routing.
        """
        self._resolver = resolver
        self._feature_flags = feature_flags
        self._router = LegacyRouter(feature_flags)

        # Pipeline components
        self._extractor = SemanticExtractor(resolver)
        self._coverage_validator = CoverageValidator(feature_flags)
        self._repair = SemanticRepair(resolver)
        self._coordinator = IntentResolutionCoordinator()
        self._preflight_loader = PreflightDataLoader()
        self._schema_service = SchemaService(resolver=resolver)
        self._candidate_generator = CandidateGenerator()
        self._column_grounder = ColumnGrounder(resolver=resolver)
        self._predicate_grounder = PredicateGrounder(resolver=resolver)
        self._canonicalizer = Canonicalizer()
        self._clarification_patcher = ClarificationDraftPatcher()

        # Lazy-loaded to avoid circular imports
        Compiler, _, _ = _import_compiler()
        Executor, _, _, _, _ = _import_executor()
        self._compiler = Compiler()
        self._executor = Executor()

        # Observability
        self._metrics = PipelineMetrics()
        self._grounding_config = GroundingConfig()

    @property
    def metrics(self) -> PipelineMetrics:
        """Access pipeline metrics for observability."""
        return self._metrics

    @property
    def feature_flags(self) -> FeatureFlags:
        """Access the current feature flags."""
        return self._feature_flags

    @property
    def router(self) -> LegacyRouter:
        """Access the legacy router for flag-based routing decisions."""
        return self._router

    async def process(
        self, prompt: str, file_path: str, file_id: str
    ) -> PipelineResult:
        """Run the full pipeline from prompt to execution result.

        Orchestrates all stages in sequence, applying feature-flag routing
        and resolution policies at each step.

        Args:
            prompt: The user's natural-language prompt.
            file_path: Path to the source data file.
            file_id: Unique identifier for the file.

        Returns:
            PipelineResult with status, canonical_intent, execution plan,
            clarification_needed, or error details.
        """
        submission_id = str(uuid.uuid4())
        warnings: list[str] = []

        tracing = PipelineTracingContext(
            submission_id=submission_id,
            pipeline_stage=PipelineStage.EXTRACTION.value,
            model_version="1.0",
        )

        # Reset repair state for this invocation
        self._repair.reset()

        # ===================================================================
        # Stage 1: Extraction
        # ===================================================================
        if self._router.route_extraction() == PipelineRoute.LEGACY_PIPELINE:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error="Legacy extraction not implemented in new pipeline",
                submission_id=submission_id,
                warnings=warnings,
            )

        tracing.pipeline_stage = PipelineStage.EXTRACTION.value
        self._metrics.record_extraction_attempt(tracing)

        try:
            draft = await self._extractor.extract(prompt)
        except ExtractionError as exc:
            self._metrics.record_extraction_failure(tracing)
            logger.error(
                "Extraction failed: %s", exc, extra=tracing.to_dict()
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Extraction failed: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )
        except (LLMProviderError, LLMValidationError) as exc:
            self._metrics.record_extraction_failure(tracing)
            logger.error(
                "LLM provider error during extraction: %s",
                exc,
                extra=tracing.to_dict(),
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Interpretation failed: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        self._metrics.record_extraction_success(tracing)
        tracing.draft_id = draft.draft_id
        tracing.draft_revision = draft.draft_revision

        # ===================================================================
        # Stage 2: Coverage Validation
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.COVERAGE_VALIDATION.value
        coverage_result: CoverageValidationResult | None = None

        if self._router.route_coverage() == PipelineRoute.NEW_PIPELINE:
            coverage_result = self._coverage_validator.validate(draft, prompt)

            if coverage_result.passed:
                self._metrics.record_coverage_check_pass(tracing)
            else:
                self._metrics.record_coverage_check_failure(tracing)

                # ===================================================================
                # Stage 3: Bounded Repair (only on coverage failure)
                # ===================================================================
                if self._router.route_repair() == PipelineRoute.NEW_PIPELINE:
                    tracing.pipeline_stage = PipelineStage.SEMANTIC_REPAIR.value
                    self._metrics.record_repair_attempt(tracing)

                    try:
                        patches = await self._repair.repair(
                            draft, coverage_result.failures, prompt
                        )
                        if patches:
                            draft = apply_patches(draft, patches)
                            tracing.draft_revision = draft.draft_revision
                            self._metrics.record_repair_success(tracing)
                    except RepairAlreadyAttemptedError:
                        pass
                    except (LLMProviderError, LLMValidationError) as exc:
                        logger.warning(
                            "Repair LLM call failed: %s", exc,
                            extra=tracing.to_dict(),
                        )

                    # Re-validate after repair
                    post_repair_result = self._coverage_validator.validate(
                        draft, prompt
                    )
                    if not post_repair_result.passed:
                        # Req 4.5: fail closed with needs_clarification after
                        # one repair attempt
                        draft = mark_needs_clarification(draft)
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason="Structural coverage gaps remain after repair",
                                element_path="coverage",
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )

        # ===================================================================
        # Stage 4: Intent Resolution Coordination
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.RESOLUTION_COORDINATION.value

        try:
            draft = self._coordinator.resolve(draft)
        except Exception as exc:
            # Req 10.3: Operation ambiguity → Clarification
            if "ambiguity" in str(exc).lower():
                return PipelineResult(
                    status=PipelineResultStatus.NEEDS_CLARIFICATION,
                    clarification_needed=ClarificationNeeded(
                        reason=f"Operation ambiguity: {exc}",
                        element_path="operation",
                        resume_directive=StageResumeDirective.VALIDATION_AND_GROUNDING,
                    ),
                    submission_id=submission_id,
                    warnings=warnings,
                )
            raise

        # Check if coordinator detected ambiguities requiring clarification
        if draft.resolution_status == ResolutionStatus.NEEDS_CLARIFICATION:
            return PipelineResult(
                status=PipelineResultStatus.NEEDS_CLARIFICATION,
                clarification_needed=ClarificationNeeded(
                    reason="Operation or scope ambiguity requires clarification",
                    element_path="operation",
                    resume_directive=StageResumeDirective.VALIDATION_AND_GROUNDING,
                ),
                submission_id=submission_id,
                warnings=warnings,
            )

        # ===================================================================
        # Stage 5: Preflight Data Load
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.PREFLIGHT_DATA_LOAD.value

        if self._router.route_grounding() == PipelineRoute.LEGACY_PIPELINE:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error="Legacy grounding not implemented in new pipeline",
                submission_id=submission_id,
                warnings=warnings,
            )

        try:
            profile, snapshot_ref = self._preflight_loader.load(
                file_path, file_id
            )
            tracing.schema_fingerprint = (
                snapshot_ref.structural_schema_fingerprint
            )
            tracing.profile_fingerprint = (
                snapshot_ref.profile_fingerprint
            )
            tracing.data_snapshot_ref = snapshot_ref.file_id
        except Exception as exc:
            logger.error(
                "Preflight data load failed: %s", exc,
                extra=tracing.to_dict(),
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Preflight data load failed: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # ===================================================================
        # Stage 6: Schema Inference
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.SCHEMA_INFERENCE.value

        try:
            schema_result = self._schema_service.infer_roles(
                snapshot_ref, profile
            )
        except Exception as exc:
            # Req 10.1: Formatting uncertainty → warning + proceed
            logger.warning(
                "Schema inference failed, proceeding with limited schema: %s",
                exc,
                extra=tracing.to_dict(),
            )
            schema_result = None
            warnings.append(f"Schema inference degraded: {exc}")

        # ===================================================================
        # Stage 7: Candidate Generation
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.COLUMN_GROUNDING.value
        self._metrics.record_grounding_attempt(tracing)

        # Collect all column references from draft actions
        standalone_refs = _extract_standalone_references(draft)
        predicate_refs = _extract_predicate_references(draft)

        # Generate candidates for all references
        candidates_by_ref: dict[str, list[ScoredCandidate]] = {}

        if schema_result is not None:
            all_refs: list[SemanticColumnReference] = list(standalone_refs) + [
                p.field_ref for p in predicate_refs
            ]
            for ref in all_refs:
                if ref.reference_text not in candidates_by_ref:
                    candidates = self._candidate_generator.generate_candidates(
                        reference=ref,
                        schema_result=schema_result,
                        profile=profile,
                    )
                    candidates_by_ref[ref.reference_text] = candidates

        # ===================================================================
        # Stage 8a: Column Grounding (standalone references)
        # ===================================================================
        column_results: list[ColumnGroundingResult] = []
        if standalone_refs:
            column_results = await self._column_grounder.ground(
                standalone_refs, candidates_by_ref, self._grounding_config
            )

            # Apply resolution policies for column grounding results
            for i, result in enumerate(column_results):
                ref = standalone_refs[i]
                if result.resolved_column is None:
                    # Check resolution policy
                    if result.confidence == 0.0 and not result.evidence:
                        # Req 10.4: No match → unsupported or Clarification
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"No physical column matches reference "
                                    f"'{ref.reference_text}'"
                                ),
                                element_path=f"column:{ref.reference_text}",
                                resume_directive=StageResumeDirective.GROUNDING,
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )
                    elif result.method == GroundingMethod.CLARIFICATION:
                        # Req 10.2: Semantic column uncertainty → Clarification
                        candidate_names = [
                            c.column_name for c in result.evidence[:5]
                        ]
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"Multiple plausible columns for "
                                    f"'{ref.reference_text}'"
                                ),
                                candidates=candidate_names,
                                element_path=f"column:{ref.reference_text}",
                                resume_directive=StageResumeDirective.GROUNDING,
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )

                    # Req 10.6: Low-confidence → clarify or fail
                    if result.confidence < self._grounding_config.confidence_threshold:
                        candidate_names = [
                            c.column_name for c in result.evidence[:5]
                        ]
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"Low-confidence grounding for "
                                    f"'{ref.reference_text}' "
                                    f"(confidence={result.confidence:.2f})"
                                ),
                                candidates=candidate_names,
                                element_path=f"column:{ref.reference_text}",
                                resume_directive=StageResumeDirective.GROUNDING,
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )
                else:
                    # Successfully resolved - update draft reference
                    ref.resolved_column = result.resolved_column

            if column_results and all(
                r.resolved_column is not None for r in column_results
            ):
                self._metrics.record_grounding_success(tracing)

        # ===================================================================
        # Stage 8b: Predicate Grounding (filter predicates)
        # ===================================================================
        predicate_results: list[PredicateGroundingResult] = []
        if predicate_refs:
            predicate_results = await self._predicate_grounder.ground(
                predicate_refs, candidates_by_ref, self._grounding_config
            )

            # Apply resolution policies for predicate grounding results
            for i, result in enumerate(predicate_results):
                pred = predicate_refs[i]
                ref_text = pred.field_ref.reference_text
                if result.resolved_column is None:
                    if result.confidence == 0.0 and not result.evidence:
                        # Req 10.4: No match → unsupported or Clarification
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"No physical column matches filter "
                                    f"reference '{ref_text}'"
                                ),
                                element_path=f"predicate:{ref_text}",
                                resume_directive=(
                                    StageResumeDirective.PREDICATE_GROUNDING
                                ),
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )
                    elif result.confidence < self._grounding_config.confidence_threshold:
                        # Req 10.6: Low-confidence → clarify or fail
                        candidate_names = [
                            c.column_name for c in result.evidence[:5]
                        ]
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"Low-confidence predicate grounding for "
                                    f"'{ref_text}' "
                                    f"(confidence={result.confidence:.2f})"
                                ),
                                candidates=candidate_names,
                                element_path=f"predicate:{ref_text}",
                                resume_directive=(
                                    StageResumeDirective.PREDICATE_GROUNDING
                                ),
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )

                    else:
                        # Req 10.2: Semantic column uncertainty → Clarification
                        candidate_names = [
                            c.column_name for c in result.evidence[:5]
                        ]
                        return PipelineResult(
                            status=PipelineResultStatus.NEEDS_CLARIFICATION,
                            clarification_needed=ClarificationNeeded(
                                reason=(
                                    f"Ambiguous filter column for '{ref_text}'"
                                ),
                                candidates=candidate_names,
                                element_path=f"predicate:{ref_text}",
                                resume_directive=(
                                    StageResumeDirective.PREDICATE_GROUNDING
                                ),
                            ),
                            submission_id=submission_id,
                            warnings=warnings,
                        )
                else:
                    # Successfully resolved - update draft predicate reference
                    pred.field_ref.resolved_column = result.resolved_column

        # Set data_snapshot_ref on draft for canonicalization
        draft.data_snapshot_ref = snapshot_ref

        # Mark draft as resolved after all grounding completes
        try:
            draft = mark_resolved(draft, ResolutionOrigin.AUTOMATIC_GROUNDING)
        except Exception as exc:
            # Req 10.5: Contract violation → fail closed
            logger.error(
                "Contract violation: cannot mark draft resolved: %s",
                exc,
                extra=tracing.to_dict(),
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Contract violation: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # ===================================================================
        # Stage 9: Canonicalization (Req 6.1 - grounding BEFORE compilation)
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.CANONICALIZATION.value

        try:
            canonical_intent = self._canonicalizer.canonicalize(draft)
        except CanonicalizeError as exc:
            # Req 10.5: Contract violation → fail closed
            logger.error(
                "Canonicalization failed (contract violation): %s",
                exc,
                extra=tracing.to_dict(),
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Contract violation at canonicalization: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        tracing.intent_id = canonical_intent.intent_id

        # ===================================================================
        # Stage 10: Compilation (Req 6.4 - no grounding during/after)
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.COMPILATION.value

        try:
            execution_plan = self._compiler.compile(canonical_intent)
        except Exception as exc:
            # Req 10.5: Contract violation → fail closed
            logger.error(
                "Compilation failed (contract violation): %s",
                exc,
                extra=tracing.to_dict(),
            )
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Compilation contract violation: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # ===================================================================
        # Stage 11: Execution
        # ===================================================================
        tracing.pipeline_stage = PipelineStage.EXECUTION.value

        # Build the intent package for the executor
        all_resolved_columns: set[str] = set()
        for step in execution_plan.steps:
            all_resolved_columns.update(step.resolved_columns)

        _, _, ExecutorIntentPackage, _, _ = _import_executor()
        intent_package = ExecutorIntentPackage(
            validated_columns=all_resolved_columns,
            data_snapshot_ref=snapshot_ref,
        )

        try:
            execution_result = self._executor.execute(
                execution_plan,
                intent_package,
                content_hash_at_execution=snapshot_ref.content_hash,
            )
        except Exception as exc:
            # Req 10.5: Contract violation → fail closed
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Execution contract violation: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # ===================================================================
        # Success: all stages completed
        # ===================================================================
        return PipelineResult(
            status=PipelineResultStatus.RESOLVED,
            canonical_intent=canonical_intent,
            execution_plan=execution_plan,
            execution_result=execution_result,
            submission_id=submission_id,
            warnings=warnings,
        )

    async def resume_after_clarification(
        self,
        draft: SemanticIntentDraft,
        clarification: ClarificationResponse,
        prompt: str,
        file_path: str,
        file_id: str,
    ) -> PipelineResult:
        """Resume pipeline from the appropriate stage after user clarification.

        Implements stage-resume policy (Req 9.3):
        - column clarification → resumes from grounding
        - operation clarification → resumes from validation + grounding
        - value clarification → resumes from predicate grounding
        - prompt replacement → restarts from extraction

        Args:
            draft: Current SemanticIntentDraft (pre-clarification).
            clarification: User's clarification response.
            prompt: The original or updated user prompt.
            file_path: Path to the source data file.
            file_id: Unique identifier for the file.

        Returns:
            PipelineResult from the resumed execution.
        """
        # Apply the clarification patch to the draft
        patched_draft, resume_directive = self._clarification_patcher.patch_draft(
            draft, clarification, draft.draft_revision
        )

        self._metrics.record_clarification_resolved(
            PipelineTracingContext(
                submission_id=str(uuid.uuid4()),
                pipeline_stage=PipelineStage.CLARIFICATION.value,
                model_version="1.0",
            )
        )

        # Route based on stage-resume directive (Req 9.3)
        if resume_directive == StageResumeDirective.EXTRACTION:
            # Full restart from extraction (prompt replacement)
            return await self.process(prompt, file_path, file_id)

        if resume_directive == StageResumeDirective.VALIDATION_AND_GROUNDING:
            # Resume from coverage validation through all remaining stages
            return await self._resume_from_validation(
                patched_draft, prompt, file_path, file_id
            )

        if resume_directive == StageResumeDirective.GROUNDING:
            # Resume from grounding stage
            return await self._resume_from_grounding(
                patched_draft, prompt, file_path, file_id
            )

        if resume_directive == StageResumeDirective.PREDICATE_GROUNDING:
            # Resume from predicate grounding only
            return await self._resume_from_grounding(
                patched_draft, prompt, file_path, file_id
            )

        # Default: restart full pipeline
        return await self.process(prompt, file_path, file_id)

    async def _resume_from_validation(
        self,
        draft: SemanticIntentDraft,
        prompt: str,
        file_path: str,
        file_id: str,
    ) -> PipelineResult:
        """Resume pipeline from coverage validation stage.

        Re-runs: coverage validation → repair → coordinator → preflight →
        schema → candidate gen → grounding → canonicalization → compilation →
        execution.
        """
        submission_id = str(uuid.uuid4())
        warnings: list[str] = []
        tracing = PipelineTracingContext(
            submission_id=submission_id,
            pipeline_stage=PipelineStage.COVERAGE_VALIDATION.value,
            model_version="1.0",
            draft_id=draft.draft_id,
            draft_revision=draft.draft_revision,
        )

        # Coverage validation on resumed draft
        if self._router.route_coverage() == PipelineRoute.NEW_PIPELINE:
            coverage_result = self._coverage_validator.validate(draft, prompt)
            if not coverage_result.passed:
                return PipelineResult(
                    status=PipelineResultStatus.NEEDS_CLARIFICATION,
                    clarification_needed=ClarificationNeeded(
                        reason="Coverage gaps after clarification resume",
                        element_path="coverage",
                    ),
                    submission_id=submission_id,
                    warnings=warnings,
                )

        # Coordinator
        try:
            draft = self._coordinator.resolve(draft)
        except Exception as exc:
            if "ambiguity" in str(exc).lower():
                return PipelineResult(
                    status=PipelineResultStatus.NEEDS_CLARIFICATION,
                    clarification_needed=ClarificationNeeded(
                        reason=f"Operation ambiguity: {exc}",
                        element_path="operation",
                        resume_directive=StageResumeDirective.VALIDATION_AND_GROUNDING,
                    ),
                    submission_id=submission_id,
                    warnings=warnings,
                )
            raise

        if draft.resolution_status == ResolutionStatus.NEEDS_CLARIFICATION:
            return PipelineResult(
                status=PipelineResultStatus.NEEDS_CLARIFICATION,
                clarification_needed=ClarificationNeeded(
                    reason="Operation ambiguity remains after clarification",
                    element_path="operation",
                    resume_directive=StageResumeDirective.VALIDATION_AND_GROUNDING,
                ),
                submission_id=submission_id,
                warnings=warnings,
            )

        # Continue from grounding onwards
        return await self._resume_from_grounding(
            draft, prompt, file_path, file_id
        )

    async def _resume_from_grounding(
        self,
        draft: SemanticIntentDraft,
        prompt: str,
        file_path: str,
        file_id: str,
    ) -> PipelineResult:
        """Resume pipeline from the grounding stage (preflight → end).

        Re-runs: preflight → schema → candidate gen → grounding →
        canonicalization → compilation → execution.
        """
        submission_id = str(uuid.uuid4())
        warnings: list[str] = []
        tracing = PipelineTracingContext(
            submission_id=submission_id,
            pipeline_stage=PipelineStage.PREFLIGHT_DATA_LOAD.value,
            model_version="1.0",
            draft_id=draft.draft_id,
            draft_revision=draft.draft_revision,
        )

        # Preflight load
        try:
            profile, snapshot_ref = self._preflight_loader.load(
                file_path, file_id
            )
        except Exception as exc:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Preflight data load failed on resume: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # Schema inference
        tracing.pipeline_stage = PipelineStage.SCHEMA_INFERENCE.value
        try:
            schema_result = self._schema_service.infer_roles(
                snapshot_ref, profile
            )
        except Exception as exc:
            logger.warning("Schema inference failed on resume: %s", exc)
            schema_result = None
            warnings.append(f"Schema inference degraded: {exc}")

        # Candidate generation
        tracing.pipeline_stage = PipelineStage.COLUMN_GROUNDING.value
        standalone_refs = _extract_standalone_references(draft)
        predicate_refs = _extract_predicate_references(draft)

        candidates_by_ref: dict[str, list[ScoredCandidate]] = {}

        if schema_result is not None:
            all_refs: list[SemanticColumnReference] = list(standalone_refs) + [
                p.field_ref for p in predicate_refs
            ]
            for ref in all_refs:
                if ref.reference_text not in candidates_by_ref:
                    candidates = self._candidate_generator.generate_candidates(
                        reference=ref,
                        schema_result=schema_result,
                        profile=profile,
                    )
                    candidates_by_ref[ref.reference_text] = candidates

        # Column grounding
        if standalone_refs:
            column_results = await self._column_grounder.ground(
                standalone_refs, candidates_by_ref, self._grounding_config
            )
            for i, result in enumerate(column_results):
                ref = standalone_refs[i]
                if result.resolved_column is None:
                    candidate_names = [
                        c.column_name for c in result.evidence[:5]
                    ]
                    return PipelineResult(
                        status=PipelineResultStatus.NEEDS_CLARIFICATION,
                        clarification_needed=ClarificationNeeded(
                            reason=(
                                f"Cannot resolve '{ref.reference_text}' "
                                f"after clarification"
                            ),
                            candidates=candidate_names,
                            element_path=f"column:{ref.reference_text}",
                            resume_directive=StageResumeDirective.GROUNDING,
                        ),
                        submission_id=submission_id,
                        warnings=warnings,
                    )
                ref.resolved_column = result.resolved_column

        # Predicate grounding
        if predicate_refs:
            predicate_results = await self._predicate_grounder.ground(
                predicate_refs, candidates_by_ref, self._grounding_config
            )
            for i, result in enumerate(predicate_results):
                pred = predicate_refs[i]
                if result.resolved_column is None:
                    candidate_names = [
                        c.column_name for c in result.evidence[:5]
                    ]
                    return PipelineResult(
                        status=PipelineResultStatus.NEEDS_CLARIFICATION,
                        clarification_needed=ClarificationNeeded(
                            reason=(
                                f"Cannot resolve filter column "
                                f"'{pred.field_ref.reference_text}' "
                                f"after clarification"
                            ),
                            candidates=candidate_names,
                            element_path=(
                                f"predicate:{pred.field_ref.reference_text}"
                            ),
                            resume_directive=(
                                StageResumeDirective.PREDICATE_GROUNDING
                            ),
                        ),
                        submission_id=submission_id,
                        warnings=warnings,
                    )
                pred.field_ref.resolved_column = result.resolved_column

        # Set data_snapshot_ref and mark resolved
        draft.data_snapshot_ref = snapshot_ref
        try:
            draft = mark_resolved(draft, ResolutionOrigin.AUTOMATIC_GROUNDING)
        except Exception as exc:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Contract violation on resume: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # Canonicalization
        tracing.pipeline_stage = PipelineStage.CANONICALIZATION.value
        try:
            canonical_intent = self._canonicalizer.canonicalize(draft)
        except CanonicalizeError as exc:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Canonicalization failed on resume: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # Compilation
        tracing.pipeline_stage = PipelineStage.COMPILATION.value
        try:
            execution_plan = self._compiler.compile(canonical_intent)
        except Exception as exc:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Compilation failed on resume: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        # Execution
        tracing.pipeline_stage = PipelineStage.EXECUTION.value
        all_resolved_columns: set[str] = set()
        for step in execution_plan.steps:
            all_resolved_columns.update(step.resolved_columns)

        _, _, ExecutorIntentPackage, _, _ = _import_executor()
        intent_package = ExecutorIntentPackage(
            validated_columns=all_resolved_columns,
            data_snapshot_ref=snapshot_ref,
        )

        try:
            execution_result = self._executor.execute(
                execution_plan,
                intent_package,
                content_hash_at_execution=snapshot_ref.content_hash,
            )
        except Exception as exc:
            return PipelineResult(
                status=PipelineResultStatus.FAILED,
                error=f"Execution contract violation on resume: {exc}",
                submission_id=submission_id,
                warnings=warnings,
            )

        return PipelineResult(
            status=PipelineResultStatus.RESOLVED,
            canonical_intent=canonical_intent,
            execution_plan=execution_plan,
            execution_result=execution_result,
            submission_id=submission_id,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_standalone_references(
    draft: SemanticIntentDraft,
) -> list[SemanticColumnReference]:
    """Extract standalone column references from draft actions.

    These are references in project, drop, sort, and rename actions
    (NOT filter predicates). Routed to Column Grounder only.
    """
    refs: list[SemanticColumnReference] = []
    for action in draft.actions:
        if isinstance(action, ProjectAction):
            for col_ref in action.columns:
                if col_ref.resolved_column is None:
                    refs.append(col_ref)
        elif isinstance(action, DropAction):
            for col_ref in action.columns:
                if col_ref.resolved_column is None:
                    refs.append(col_ref)
        elif isinstance(action, SortAction):
            for col_ref in action.keys:
                if col_ref.resolved_column is None:
                    refs.append(col_ref)
        elif isinstance(action, RenameAction):
            for col_ref, _new_name in action.mappings:
                if col_ref.resolved_column is None:
                    refs.append(col_ref)
    return refs


def _extract_predicate_references(
    draft: SemanticIntentDraft,
) -> list[UnresolvedPredicate]:
    """Extract filter predicate references from draft actions.

    These are field+operator+value predicates inside filter actions.
    Routed to Predicate Grounder only.
    """
    predicates: list[UnresolvedPredicate] = []
    for action in draft.actions:
        if isinstance(action, FilterAction):
            for group in action.logical_groups:
                for predicate in group.predicates:
                    if predicate.field_ref.resolved_column is None:
                        predicates.append(predicate)
    return predicates
