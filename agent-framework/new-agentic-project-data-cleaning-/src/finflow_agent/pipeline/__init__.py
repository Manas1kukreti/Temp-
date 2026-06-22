"""Pipeline orchestration package for FinFlow's semantic pipeline.

This package contains the pipeline infrastructure components that coordinate
the multi-stage intent resolution process from raw prompt to CanonicalIntent.

Key components:
- coordinator: Intent Resolution Coordinator (operation classification, boolean scope)
- clarification_service: Draft-patching extension with stage-resume policy
- coverage_validator: Deterministic Coverage Validator (structural checks, provenance
  completeness, shadow LLM mode)
- semantic_repair: Bounded Semantic Repair (one attempt, typed patches only)
- canonicalizer: Draft to CanonicalIntent type-level boundary enforcement
- feature_flags: Feature flag system with compatibility matrix validation
- observability: Structured tracing, metrics, and shadow comparison recording
"""

from finflow_agent.pipeline.canonicalizer import Canonicalizer
from finflow_agent.pipeline.clarification_service import (
    ClarificationDraftPatcher,
    ClarificationResponse,
    DuplicateIdempotencyKeyError,
    PatchType,
    StageResumeDirective,
    StaleRevisionError,
)
from finflow_agent.pipeline.coordinator import (
    IntentResolutionCoordinator,
    OperationAmbiguityError,
)
from finflow_agent.pipeline.coverage_validator import (
    CoverageValidationResult,
    CoverageValidator,
    StructuralFailure,
)
from finflow_agent.pipeline.observability import (
    DecisionOwnerRecord,
    DecisionOwnerRecorder,
    JSONFormatter,
    MetricEvent,
    PipelineMetrics,
    PipelineStage,
    PipelineTracingContext,
    ShadowModeRecorder,
)
from finflow_agent.pipeline.patch_applicator import (
    PatchApplicationError,
    apply_patches,
)
from finflow_agent.pipeline.resolution_handler import (
    ResolutionError,
    mark_interpretation_failed,
    mark_needs_clarification,
    mark_resolved,
    set_resolution_origin,
    set_resolution_status,
)
from finflow_agent.pipeline.legacy_router import (
    LegacyRouter,
    PipelineRoute,
)
from finflow_agent.pipeline.orchestrator import (
    ClarificationNeeded,
    PipelineResult,
    PipelineResultStatus,
    SemanticPipeline,
)
from finflow_agent.pipeline.semantic_repair import (
    RepairAlreadyAttemptedError,
    SemanticRepair,
)

__all__ = [
    "Canonicalizer",
    "ClarificationDraftPatcher",
    "ClarificationNeeded",
    "ClarificationResponse",
    "CoverageValidationResult",
    "CoverageValidator",
    "DecisionOwnerRecord",
    "DecisionOwnerRecorder",
    "DuplicateIdempotencyKeyError",
    "IntentResolutionCoordinator",
    "JSONFormatter",
    "LegacyRouter",
    "MetricEvent",
    "OperationAmbiguityError",
    "PatchApplicationError",
    "PatchType",
    "PipelineMetrics",
    "PipelineResult",
    "PipelineResultStatus",
    "PipelineRoute",
    "PipelineStage",
    "PipelineTracingContext",
    "RepairAlreadyAttemptedError",
    "ResolutionError",
    "SemanticPipeline",
    "ShadowModeRecorder",
    "SemanticRepair",
    "StageResumeDirective",
    "StaleRevisionError",
    "StructuralFailure",
    "apply_patches",
    "mark_interpretation_failed",
    "mark_needs_clarification",
    "mark_resolved",
    "set_resolution_origin",
    "set_resolution_status",
]
