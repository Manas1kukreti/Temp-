"""Resolution status and origin handling for FinFlow's semantic pipeline.

Provides helper functions for managing resolution_status (validity) and
resolution_origin (workflow path) transitions on SemanticIntentDraft objects.
Each transition produces a new immutable draft revision (N → N+1).

Key invariant: resolution_status tracks validity lifecycle independently of
resolution_origin which tracks the workflow path. The Compiler accepts
CanonicalIntent regardless of resolution_origin (Req 20.4).

Functions:
- set_resolution_status: update status on a draft, producing a new revision
- set_resolution_origin: update origin on a draft, producing a new revision
- mark_resolved: convenience for setting RESOLVED + an origin (with validation)
- mark_needs_clarification: convenience for NEEDS_CLARIFICATION status
- mark_interpretation_failed: convenience for INTERPRETATION_FAILED (provider errors)

Requirements: 20.1, 20.3, 20.5
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from finflow_agent.models.draft import (
    FilterAction,
    ProjectAction,
    DropAction,
    RenameAction,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
)


class ResolutionError(Exception):
    """Raised when a resolution transition is invalid.

    For example, marking a draft as RESOLVED when it still contains
    ungrounded column references.
    """

    pass


def _new_revision(draft: SemanticIntentDraft, **overrides: object) -> SemanticIntentDraft:
    """Create a new draft revision with specified field overrides.

    Deep-copies the draft, increments draft_revision, and applies overrides.
    The original draft is never mutated (Req 17.1).
    """
    draft_dict = copy.deepcopy(draft.model_dump(mode="json"))
    draft_dict["draft_revision"] = draft.draft_revision + 1
    draft_dict["created_at"] = datetime.now(timezone.utc).isoformat()

    for key, value in overrides.items():
        if isinstance(value, ResolutionStatus):
            draft_dict[key] = value.value
        elif isinstance(value, ResolutionOrigin):
            draft_dict[key] = value.value
        else:
            draft_dict[key] = value

    return SemanticIntentDraft.model_validate_json(json.dumps(draft_dict))


def _all_columns_grounded(draft: SemanticIntentDraft) -> bool:
    """Check that every SemanticColumnReference in the draft has a resolved_column set.

    This is a prerequisite for marking a draft as RESOLVED.
    """
    for action in draft.actions:
        if isinstance(action, FilterAction):
            for group in action.logical_groups:
                for predicate in group.predicates:
                    if predicate.field_ref.resolved_column is None:
                        return False
        elif isinstance(action, ProjectAction):
            for col_ref in action.columns:
                if col_ref.resolved_column is None:
                    return False
        elif isinstance(action, DropAction):
            for col_ref in action.columns:
                if col_ref.resolved_column is None:
                    return False
        elif isinstance(action, SortAction):
            for col_ref in action.keys:
                if col_ref.resolved_column is None:
                    return False
        elif isinstance(action, RenameAction):
            for col_ref, _new_name in action.mappings:
                if col_ref.resolved_column is None:
                    return False
    return True


def set_resolution_status(
    draft: SemanticIntentDraft, status: ResolutionStatus
) -> SemanticIntentDraft:
    """Create a new draft revision with the updated resolution_status.

    The resolution_status tracks the validity lifecycle of the draft,
    independently of the workflow path (resolution_origin).

    Args:
        draft: The current SemanticIntentDraft (remains immutable).
        status: The new ResolutionStatus to set.

    Returns:
        A new SemanticIntentDraft at revision N+1 with updated status.

    Requirements: 20.1
    """
    return _new_revision(draft, resolution_status=status)


def set_resolution_origin(
    draft: SemanticIntentDraft, origin: ResolutionOrigin
) -> SemanticIntentDraft:
    """Create a new draft revision with the specified resolution_origin.

    The resolution_origin reflects the actual workflow path that produced
    the resolution, independent of status transitions (Req 20.5).

    Args:
        draft: The current SemanticIntentDraft (remains immutable).
        origin: The ResolutionOrigin indicating the workflow path.

    Returns:
        A new SemanticIntentDraft at revision N+1 with updated origin.

    Requirements: 20.5
    """
    return _new_revision(draft, resolution_origin=origin)


def mark_resolved(
    draft: SemanticIntentDraft, origin: ResolutionOrigin
) -> SemanticIntentDraft:
    """Convenience: set status=RESOLVED and the given origin.

    Validates that all column references in the draft are grounded before
    allowing the RESOLVED status. This ensures the key invariant that a
    resolved draft has no unresolved active execution references.

    Args:
        draft: The current SemanticIntentDraft (remains immutable).
        origin: The ResolutionOrigin reflecting how resolution was achieved.

    Returns:
        A new SemanticIntentDraft at revision N+1 with RESOLVED status and origin.

    Raises:
        ResolutionError: If the draft contains ungrounded column references.

    Requirements: 20.1, 20.3, 20.5
    """
    if not _all_columns_grounded(draft):
        raise ResolutionError(
            "Cannot mark draft as RESOLVED: one or more column references "
            "are not grounded to physical columns. All SemanticColumnReference "
            "objects must have resolved_column set before resolution."
        )

    return _new_revision(
        draft,
        resolution_status=ResolutionStatus.RESOLVED,
        resolution_origin=origin,
    )


def mark_needs_clarification(draft: SemanticIntentDraft) -> SemanticIntentDraft:
    """Set status=NEEDS_CLARIFICATION on the draft.

    Used when the pipeline encounters genuine user ambiguity that requires
    interactive clarification (not provider errors).

    Args:
        draft: The current SemanticIntentDraft (remains immutable).

    Returns:
        A new SemanticIntentDraft at revision N+1 with NEEDS_CLARIFICATION status.

    Requirements: 20.1
    """
    return _new_revision(draft, resolution_status=ResolutionStatus.NEEDS_CLARIFICATION)


def mark_interpretation_failed(draft: SemanticIntentDraft) -> SemanticIntentDraft:
    """Set status=INTERPRETATION_FAILED on the draft.

    Used for provider errors (timeout, rate limit, invalid JSON, schema validation
    failure, empty output) — NOT for genuine user ambiguity. This distinction
    ensures the system does not ask users questions arising from system errors.

    Args:
        draft: The current SemanticIntentDraft (remains immutable).

    Returns:
        A new SemanticIntentDraft at revision N+1 with INTERPRETATION_FAILED status.

    Requirements: 18.6, 20.1
    """
    return _new_revision(
        draft, resolution_status=ResolutionStatus.INTERPRETATION_FAILED
    )
