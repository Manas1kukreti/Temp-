"""Shared data models for FinFlow's semantic pipeline.

This package defines the Pydantic models that serve as typed contracts between
pipeline stages. All models use strict configuration to prevent silent coercion.

Key components:
- draft: SemanticIntentDraft, SemanticColumnReference, action types, and enums
- canonical: CanonicalIntent (frozen, always resolved)
- provenance: ProvenanceRef discriminated union (PromptSpan, Clarification, SchemaEvidence)
- envelope: IntentEnvelope container (draft or canonical + pipeline metadata)
- patches: SemanticPatch operations (add, replace, remove)
- fingerprints: StructuralSchemaFingerprint and ProfileFingerprint (deterministic hashing)
- snapshot: DataSnapshotRef (immutable reference to profiled file version)
"""

from finflow_agent.models.canonical import (
    CanonicalIntent,
    CanonicalizeError,
    ResolvedAction,
    ResolvedDropAction,
    ResolvedFilterAction,
    ResolvedProjectAction,
    ResolvedRenameAction,
    ResolvedSortAction,
)
from finflow_agent.models.envelope import (
    IntentEnvelope,
    PipelineStatus,
    ResolutionRecord,
    ShadowComparisonMetric,
)
from finflow_agent.models.fingerprints import (
    ProfileFingerprint,
    StructuralSchemaFingerprint,
)
from finflow_agent.models.patches import PatchOp, SemanticPatch
from finflow_agent.models.provenance import (
    ClarificationProvenance,
    PromptSpanProvenance,
    ProvenanceRef,
    SchemaEvidenceProvenance,
)
from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.models.draft import (
    AmbiguityMarker,
    DraftAction,
    DropAction,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    ReferenceKind,
    RenameAction,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
    UnresolvedPredicate,
)
from finflow_agent.models.pretty_printer import pretty_print_draft
from finflow_agent.models.upcasters import (
    SUPPORTED_VERSIONS,
    UpcasterError,
    upcast_canonical_intent,
)

__all__ = [
    "AmbiguityMarker",
    "CanonicalIntent",
    "CanonicalizeError",
    "ClarificationProvenance",
    "DataSnapshotRef",
    "DraftAction",
    "DropAction",
    "FilterAction",
    "IntentEnvelope",
    "LogicalGroup",
    "PatchOp",
    "PipelineStatus",
    "ProfileFingerprint",
    "ProjectAction",
    "PromptSpanProvenance",
    "pretty_print_draft",
    "ProvenanceRef",
    "ReferenceKind",
    "RenameAction",
    "ResolvedAction",
    "ResolvedDropAction",
    "ResolvedFilterAction",
    "ResolvedProjectAction",
    "ResolvedRenameAction",
    "ResolvedSortAction",
    "ResolutionOrigin",
    "ResolutionRecord",
    "ResolutionStatus",
    "SUPPORTED_VERSIONS",
    "SchemaEvidenceProvenance",
    "SemanticColumnReference",
    "SemanticIntentDraft",
    "SemanticPatch",
    "ShadowComparisonMetric",
    "SortAction",
    "StructuralSchemaFingerprint",
    "UnresolvedPredicate",
    "UpcasterError",
    "upcast_canonical_intent",
]
