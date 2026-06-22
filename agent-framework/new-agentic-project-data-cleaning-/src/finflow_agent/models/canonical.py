"""CanonicalIntent and resolved action models for FinFlow's semantic pipeline.

Defines the fully-resolved, immutable, executable intent representation.
A CanonicalIntent can only be constructed from a SemanticIntentDraft that has
resolution_status == RESOLVED and all column references grounded to physical names.

Key types:
- CanonicalizeError: raised when a draft cannot be promoted to canonical
- ResolvedAction: base for resolved action types (frozen)
- ResolvedFilterAction, ResolvedProjectAction, ResolvedDropAction,
  ResolvedSortAction, ResolvedRenameAction: concrete resolved action types
- CanonicalIntent: the finalized, frozen, always-resolved intent

Requirements: 6.2, 11.2, 20.2, 20.4, 21.1, 21.2, 21.4, 21.5
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.draft import (
    DraftAction,
    DropAction,
    FilterAction,
    ProjectAction,
    RenameAction,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
)
from finflow_agent.models.envelope import ResolutionRecord
from finflow_agent.models.provenance import ProvenanceRef
from finflow_agent.models.snapshot import DataSnapshotRef


class CanonicalizeError(Exception):
    """Raised when a draft cannot be canonicalized.

    This occurs when:
    - The draft's resolution_status is not RESOLVED
    - Any SemanticColumnReference in the draft has resolved_column == None

    Requirements: 6.2
    """

    pass


# ---------------------------------------------------------------------------
# Resolved action types — all frozen, all columns are physical names
# ---------------------------------------------------------------------------


class ResolvedFilterAction(BaseModel):
    """A filter action where all column references are resolved to physical names.

    Requirements: 21.5 - frozen ensures immutability after construction.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    type: Literal["filter"] = "filter"
    predicates: list[dict[str, object]] = Field(
        ...,
        min_length=1,
        description="Resolved predicates with physical column names, operators, and values",
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the filter action"
    )


class ResolvedProjectAction(BaseModel):
    """A project action where all columns are resolved to physical names.

    Requirements: 21.5 - frozen ensures immutability after construction.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    type: Literal["project"] = "project"
    columns: list[str] = Field(
        ..., min_length=1, description="Physical column names to project/select"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the project action"
    )


class ResolvedDropAction(BaseModel):
    """A drop action where all columns are resolved to physical names.

    Requirements: 21.5 - frozen ensures immutability after construction.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    type: Literal["drop"] = "drop"
    columns: list[str] = Field(
        ..., min_length=1, description="Physical column names to drop"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the drop action"
    )


class ResolvedSortAction(BaseModel):
    """A sort action where all columns are resolved to physical names.

    Requirements: 21.5 - frozen ensures immutability after construction.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    type: Literal["sort"] = "sort"
    keys: list[str] = Field(
        ..., min_length=1, description="Physical column names to sort by"
    )
    directions: list[Literal["asc", "desc"]] = Field(
        ..., min_length=1, description="Sort direction for each key"
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the sort action"
    )


class ResolvedRenameAction(BaseModel):
    """A rename action where all source columns are resolved to physical names.

    Requirements: 21.5 - frozen ensures immutability after construction.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    type: Literal["rename"] = "rename"
    mappings: list[tuple[str, str]] = Field(
        ...,
        min_length=1,
        description="List of (physical_source_column, new_name) pairs",
    )
    provenance: list[ProvenanceRef] = Field(
        ..., min_length=1, description="Provenance for the rename action"
    )


ResolvedAction = Annotated[
    Union[
        ResolvedFilterAction,
        ResolvedProjectAction,
        ResolvedDropAction,
        ResolvedSortAction,
        ResolvedRenameAction,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all resolved action types.

Uses Pydantic's discriminator on the 'type' field for unambiguous deserialization.
All resolved actions contain only physical column names (no unresolved references).
"""


# ---------------------------------------------------------------------------
# CanonicalIntent — the fully resolved, frozen model
# ---------------------------------------------------------------------------


class CanonicalIntent(BaseModel):
    """Fully resolved, executable intent.

    Can only be constructed from a resolved SemanticIntentDraft via the
    from_resolved_draft() factory method. Contains no unresolved active
    execution references. The frozen=True config ensures immutability
    after construction.

    Requirements: 6.2, 11.2, 20.2, 20.4, 21.1, 21.2, 21.4, 21.5
    """

    model_config = ConfigDict(strict=True, frozen=True)

    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: str = "1.0"

    # Always resolved — type-level guarantee (Req 20.2, 11.2)
    resolution_status: Literal["resolved"] = "resolved"
    resolution_origin: ResolutionOrigin

    # Resolved content (all columns are physical)
    actions: list[ResolvedAction]

    # Immutable audit fields (historical metadata, not active references)
    source_draft_id: str
    source_draft_revision: int
    resolution_history: list[ResolutionRecord] = Field(default_factory=list)
    provenance: list[ProvenanceRef] = Field(default_factory=list)

    # Data context
    data_snapshot_ref: DataSnapshotRef

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_resolved_draft(cls, draft: SemanticIntentDraft) -> CanonicalIntent:
        """Factory: constructs a CanonicalIntent only if the draft is fully resolved.

        Validates:
        - draft.resolution_status == RESOLVED
        - All SemanticColumnReferences in actions have resolved_column set
        - draft.resolution_origin is set
        - draft.data_snapshot_ref is set

        Raises CanonicalizeError if any validation fails.

        Requirements: 21.2, 21.4, 6.2
        """
        if draft.resolution_status != ResolutionStatus.RESOLVED:
            raise CanonicalizeError(
                f"Cannot canonicalize draft with status '{draft.resolution_status.value}'. "
                f"Draft must have resolution_status == RESOLVED."
            )

        if draft.resolution_origin is None:
            raise CanonicalizeError(
                "Cannot canonicalize draft without a resolution_origin."
            )

        if draft.data_snapshot_ref is None:
            raise CanonicalizeError(
                "Cannot canonicalize draft without a data_snapshot_ref."
            )

        # Validate all column references are resolved and convert to resolved actions
        resolved_actions: list[
            ResolvedFilterAction
            | ResolvedProjectAction
            | ResolvedDropAction
            | ResolvedSortAction
            | ResolvedRenameAction
        ] = []

        for action in draft.actions:
            resolved_action = _resolve_action(action)
            resolved_actions.append(resolved_action)

        return cls(
            resolution_origin=draft.resolution_origin,
            actions=resolved_actions,
            source_draft_id=draft.draft_id,
            source_draft_revision=draft.draft_revision,
            resolution_history=draft.resolution_history,
            provenance=draft.extraction_provenance,
            data_snapshot_ref=draft.data_snapshot_ref,
        )


# ---------------------------------------------------------------------------
# Internal helpers for action resolution
# ---------------------------------------------------------------------------


def _validate_column_resolved(ref: SemanticColumnReference, context: str) -> str:
    """Validate that a SemanticColumnReference has resolved_column set.

    Returns the resolved physical column name.
    Raises CanonicalizeError if unresolved.
    """
    if ref.resolved_column is None:
        raise CanonicalizeError(
            f"Unresolved column reference in {context}: "
            f"reference_text='{ref.reference_text}', "
            f"reference_kind='{ref.reference_kind.value}'. "
            f"All column references must be grounded before canonicalization."
        )
    return ref.resolved_column


def _resolve_action(
    action: DraftAction,
) -> (
    ResolvedFilterAction
    | ResolvedProjectAction
    | ResolvedDropAction
    | ResolvedSortAction
    | ResolvedRenameAction
):
    """Convert a draft action to its resolved counterpart.

    Validates all column references are resolved and extracts physical names.
    Raises CanonicalizeError for unresolved references.
    """
    if isinstance(action, FilterAction):
        return _resolve_filter_action(action)
    elif isinstance(action, ProjectAction):
        return _resolve_project_action(action)
    elif isinstance(action, DropAction):
        return _resolve_drop_action(action)
    elif isinstance(action, SortAction):
        return _resolve_sort_action(action)
    elif isinstance(action, RenameAction):
        return _resolve_rename_action(action)
    else:
        raise CanonicalizeError(f"Unknown action type: {type(action)}")


def _resolve_filter_action(action: FilterAction) -> ResolvedFilterAction:
    """Resolve a FilterAction, validating all predicate column references."""
    resolved_predicates: list[dict[str, object]] = []

    for logical_group in action.logical_groups:
        for predicate in logical_group.predicates:
            physical_col = _validate_column_resolved(
                predicate.field_ref, "filter predicate"
            )
            resolved_predicates.append(
                {
                    "column": physical_col,
                    "operator": predicate.operator,
                    "value": predicate.value,
                    "negated": predicate.negated,
                    "logical_operator": logical_group.operator,
                }
            )

    return ResolvedFilterAction(
        predicates=resolved_predicates,
        provenance=action.provenance,
    )


def _resolve_project_action(action: ProjectAction) -> ResolvedProjectAction:
    """Resolve a ProjectAction, validating all column references."""
    resolved_columns = [
        _validate_column_resolved(col_ref, "project action")
        for col_ref in action.columns
    ]
    return ResolvedProjectAction(
        columns=resolved_columns,
        provenance=action.provenance,
    )


def _resolve_drop_action(action: DropAction) -> ResolvedDropAction:
    """Resolve a DropAction, validating all column references."""
    resolved_columns = [
        _validate_column_resolved(col_ref, "drop action")
        for col_ref in action.columns
    ]
    return ResolvedDropAction(
        columns=resolved_columns,
        provenance=action.provenance,
    )


def _resolve_sort_action(action: SortAction) -> ResolvedSortAction:
    """Resolve a SortAction, validating all column references."""
    resolved_keys = [
        _validate_column_resolved(col_ref, "sort action")
        for col_ref in action.keys
    ]
    return ResolvedSortAction(
        keys=resolved_keys,
        directions=action.directions,
        provenance=action.provenance,
    )


def _resolve_rename_action(action: RenameAction) -> ResolvedRenameAction:
    """Resolve a RenameAction, validating all source column references."""
    resolved_mappings: list[tuple[str, str]] = []
    for col_ref, new_name in action.mappings:
        physical_col = _validate_column_resolved(col_ref, "rename action")
        resolved_mappings.append((physical_col, new_name))

    return ResolvedRenameAction(
        mappings=resolved_mappings,
        provenance=action.provenance,
    )
