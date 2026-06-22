"""SemanticIntentDraft and supporting models for FinFlow's semantic pipeline.

Defines the pre-canonical intermediate representation that preserves ambiguities,
unresolved references, and typed provenance. Each draft is an immutable revision;
modifications produce a new revision with an incremented draft_revision.

Key types:
- ReferenceKind: classification of how a column is referenced in user prompt
- ResolutionStatus: validity state of a draft (pending → resolved lifecycle)
- ResolutionOrigin: workflow path that produced the current resolution
- SemanticColumnReference: typed column reference with provenance
- DraftAction: discriminated union of pipeline actions (filter, project, drop, sort, rename)
- SemanticIntentDraft: the complete pre-canonical model

Requirements: 1.1, 1.2, 1.5, 14.1, 14.2, 14.4, 14.5, 14.6, 17.1, 20.1
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.provenance import (
    PromptSpanProvenance,
    ProvenanceRef,
)
from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.models.envelope import ResolutionRecord


class ReferenceKind(str, Enum):
    """Classification of how a column is referenced in a user prompt.

    Requirements: 1.2 - reference_kind classified as one of these values.
    """

    EXPLICIT_NAME = "explicit_name"
    SEMANTIC_CONCEPT = "semantic_concept"
    GENERIC_REFERENCE = "generic_reference"
    VALUE_IMPLIED = "value_implied"
    COLUMN_GROUP = "column_group"


class ResolutionStatus(str, Enum):
    """Validity state of a draft or intent.

    Requirements: 20.1 - resolution_status tracks validity lifecycle.
    """

    PENDING = "pending"
    NEEDS_CLARIFICATION = "needs_clarification"
    INTERPRETATION_FAILED = "interpretation_failed"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    RESOLVED = "resolved"


class ResolutionOrigin(str, Enum):
    """Workflow path that produced the current resolution.

    Requirements: 20.1 - resolution_origin reflects actual workflow path.
    """

    DIRECT = "direct"
    SEMANTIC_REPAIR = "semantic_repair"
    AUTOMATIC_GROUNDING = "automatic_grounding"
    USER_CLARIFICATION = "user_clarification"


# ---------------------------------------------------------------------------
# Core reference and predicate models
# ---------------------------------------------------------------------------


class SemanticColumnReference(BaseModel):
    """Typed column reference with reference_kind classification and provenance.

    Every column mention in the draft is represented as a SemanticColumnReference,
    classifying how the user referred to the column and retaining provenance.

    Requirements: 1.2, 1.6, 14.6
    """

    model_config = ConfigDict(strict=True)

    reference_text: str = Field(
        ..., min_length=1, description="Original text used to reference the column"
    )
    reference_kind: ReferenceKind = Field(
        ..., description="Classification of how the column is referenced"
    )
    resolved_column: str | None = Field(
        default=None, description="Physical column name after grounding (None if unresolved)"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Grounding confidence score"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="At least one provenance reference (Req 1.6)"
    )


class AmbiguityMarker(BaseModel):
    """Marks an ambiguous element with candidate resolutions.

    Requirements: 1.4 - preserve all plausible interpretations.
    """

    model_config = ConfigDict(strict=True)

    element_path: str = Field(
        ..., min_length=1, description="JSON path to the ambiguous element in the draft"
    )
    candidates: list[str] = Field(
        ..., min_length=1, description="Candidate resolutions for this ambiguity"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the ambiguity source"
    )


class UnresolvedPredicate(BaseModel):
    """A filter predicate that may not yet be grounded to a physical column.

    Requirements: 1.5 - preserve boolean scope as value set within one predicate.
    """

    model_config = ConfigDict(strict=True)

    field_ref: SemanticColumnReference = Field(
        ..., description="Semantic column reference for the predicate field"
    )
    operator: str = Field(
        ..., min_length=1, description="Predicate operator (e.g., 'eq', 'gt', 'in')"
    )
    value: Any = Field(..., description="Predicate value (may be a list for 'in' operators)")
    negated: bool = Field(default=False, description="Whether this predicate is negated")
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for this predicate"
    )


class LogicalGroup(BaseModel):
    """A group of predicates connected by a boolean operator.

    Requirements: 1.5 - preserve boolean scope.
    """

    model_config = ConfigDict(strict=True)

    operator: Literal["and", "or"] = Field(
        ..., description="Boolean operator connecting predicates"
    )
    predicates: list[UnresolvedPredicate] = Field(
        ..., min_length=1, description="Predicates in this logical group"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the logical group structure"
    )


# ---------------------------------------------------------------------------
# Discriminated action union
# ---------------------------------------------------------------------------


class FilterAction(BaseModel):
    """Action representing a filter/where operation.

    Requirements: 14.5 - discriminated union with type field.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["filter"] = "filter"
    logical_groups: list[LogicalGroup] = Field(
        ..., min_length=1, description="Logical groups defining the filter conditions"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the filter action"
    )


class ProjectAction(BaseModel):
    """Action representing a column projection/selection.

    Requirements: 14.5 - discriminated union with type field.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["project"] = "project"
    columns: list[SemanticColumnReference] = Field(
        ..., min_length=1, description="Columns to project/select"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the project action"
    )


class DropAction(BaseModel):
    """Action representing dropping columns.

    Requirements: 14.5 - discriminated union with type field.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["drop"] = "drop"
    columns: list[SemanticColumnReference] = Field(
        ..., min_length=1, description="Columns to drop"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the drop action"
    )


class SortAction(BaseModel):
    """Action representing a sort operation.

    Requirements: 14.5 - discriminated union with type field.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["sort"] = "sort"
    keys: list[SemanticColumnReference] = Field(
        ..., min_length=1, description="Columns to sort by"
    )
    directions: list[Literal["asc", "desc"]] = Field(
        ..., min_length=1, description="Sort direction for each key"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the sort action"
    )


class RenameAction(BaseModel):
    """Action representing column renaming.

    Requirements: 14.5 - discriminated union with type field.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["rename"] = "rename"
    mappings: list[tuple[SemanticColumnReference, str]] = Field(
        ..., min_length=1, description="List of (source_reference, new_name) pairs"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the rename action"
    )


DraftAction = Annotated[
    Union[FilterAction, ProjectAction, DropAction, SortAction, RenameAction],
    Field(discriminator="type"),
]
"""Discriminated union of all draft action types.

Uses Pydantic's discriminator on the 'type' field for unambiguous deserialization.
Requirements: 14.5
"""


# ---------------------------------------------------------------------------
# SemanticIntentDraft — the main pre-canonical model
# ---------------------------------------------------------------------------


class SemanticIntentDraft(BaseModel):
    """Pre-canonical intermediate representation preserving ambiguity and provenance.

    Each draft is an immutable revision. Modifications produce a new revision with
    an incremented draft_revision (monotonically increasing, starting at 1).

    The draft holds all extracted semantic content, unresolved references, ambiguity
    markers, and resolution state. Only when fully resolved can it be promoted to
    a CanonicalIntent.

    Requirements: 1.1, 1.2, 1.5, 14.1, 14.2, 14.4, 14.5, 14.6, 17.1, 20.1
    """

    model_config = ConfigDict(strict=True)

    # --- Versioning ---
    schema_version: str = Field(
        default="1.0", description="Schema version for forward-compatibility checks (Req 14.4)"
    )
    draft_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this draft lineage",
    )
    draft_revision: int = Field(
        default=1,
        ge=1,
        description="Monotonically incrementing revision number (Req 17.1)",
    )

    # --- Content ---
    raw_prompt: str = Field(
        ..., min_length=1, description="Original user prompt text"
    )
    actions: list[DraftAction] = Field(
        ..., description="Extracted actions using discriminated union (Req 14.5)"
    )
    ambiguities: list[AmbiguityMarker] = Field(
        default_factory=list,
        description="Ambiguity markers for elements with multiple plausible interpretations",
    )
    ignored_spans: list[PromptSpanProvenance] = Field(
        default_factory=list,
        description="Prompt spans that were explicitly ignored during extraction",
    )

    # --- Resolution state ---
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.PENDING,
        description="Current validity state of the draft (Req 20.1)",
    )
    resolution_origin: ResolutionOrigin | None = Field(
        default=None,
        description="Workflow path that produced current resolution (Req 20.1)",
    )

    # --- Provenance ---
    extraction_provenance: list[ProvenanceRef] = Field(
        default_factory=list,
        description="Provenance references for the overall extraction process",
    )

    # --- Metadata ---
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this revision was created",
    )
    data_snapshot_ref: DataSnapshotRef | None = Field(
        default=None,
        description="Reference to the profiled file version used for grounding",
    )

    # --- Resolution history (immutable audit trail) ---
    resolution_history: list[ResolutionRecord] = Field(
        default_factory=list,
        description="Immutable audit trail of resolution decisions",
    )
