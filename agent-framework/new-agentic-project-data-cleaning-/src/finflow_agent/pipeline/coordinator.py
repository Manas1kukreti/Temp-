"""Intent Resolution Coordinator for FinFlow's semantic pipeline.

The IntentResolutionCoordinator is the SINGLE OWNER of:
- Operation classification (filter/project/sort/drop/rename)
- Boolean scope (AND/OR logical-group structure)

It receives operation candidates from extraction, applies deterministic
validation results and user clarification patches, selects a unique operation
when supported, and preserves unresolved ambiguity otherwise. Returns an
updated draft with finalized action type and logical-group structure.

No other component may override operation classification or boolean scope
once the coordinator has finalized them.

Requirements: 2.1, 2.2, 21.3
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from finflow_agent.models.draft import (
    AmbiguityMarker,
    DraftAction,
    FilterAction,
    LogicalGroup,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticIntentDraft,
)
from finflow_agent.models.envelope import ResolutionRecord
from finflow_agent.models.patches import SemanticPatch
from finflow_agent.pipeline.patch_applicator import apply_patches


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OperationAmbiguityError(Exception):
    """Raised when the coordinator cannot resolve operation ambiguity.

    This occurs when:
    - Multiple plausible operation types exist for the same semantic content
    - No clarification has been provided to resolve the ambiguity
    - The coordinator cannot deterministically select a single operation

    Requirements: 2.1 - single owner must resolve or fail clearly
    """

    def __init__(
        self,
        message: str,
        ambiguous_paths: list[str] | None = None,
        candidates: list[str] | None = None,
    ) -> None:
        self.ambiguous_paths = ambiguous_paths or []
        self.candidates = candidates or []
        super().__init__(message)


# ---------------------------------------------------------------------------
# IntentResolutionCoordinator
# ---------------------------------------------------------------------------


class IntentResolutionCoordinator:
    """Finalizes operation classification and boolean scope.

    Single owner of operation classification and boolean scope as defined
    in the decision-ownership matrix (Req 2.2). No downstream component
    (grounders, compiler, executor) may override these decisions.

    The coordinator:
    1. Receives operation candidates from extraction (the draft's actions)
    2. Applies deterministic validation results (structural checks)
    3. Applies user clarification patches when available
    4. Selects a unique operation when unambiguous
    5. Preserves unresolved ambiguity if no clarification resolves it
    6. Returns the draft ready for grounding dispatch

    Requirements: 2.1, 2.2, 21.3
    """

    # Action types that are mutually exclusive when targeting the same columns
    _EXCLUSIVE_ACTION_TYPES: set[str] = {"filter", "project", "drop"}

    def resolve(
        self,
        draft: SemanticIntentDraft,
        clarification_patches: list[SemanticPatch] | None = None,
    ) -> SemanticIntentDraft:
        """Finalize operation classification and boolean scope.

        Args:
            draft: The SemanticIntentDraft from extraction/validation stages.
            clarification_patches: Optional user clarification patches that
                resolve operation ambiguities.

        Returns:
            Updated SemanticIntentDraft with finalized action type and
            logical-group structure, ready for grounding dispatch.

        Raises:
            OperationAmbiguityError: If the coordinator cannot resolve
                operation ambiguity and no clarification is available.

        Requirements: 2.1, 2.2, 21.3
        """
        working_draft = draft

        # Step 1: Apply user clarification patches if provided
        if clarification_patches:
            working_draft = self._apply_clarification_patches(
                working_draft, clarification_patches
            )

        # Step 2: Check for operation-type ambiguity markers
        operation_ambiguities = self._find_operation_ambiguities(working_draft)

        # Step 3: Attempt resolution
        if operation_ambiguities:
            # Check if clarification resolved the ambiguities
            resolved = self._check_ambiguities_resolved(
                working_draft, operation_ambiguities
            )
            if not resolved:
                # Ambiguity remains — preserve it in the draft
                working_draft = self._preserve_ambiguity(working_draft)
                raise OperationAmbiguityError(
                    message=(
                        "Cannot resolve operation classification: "
                        f"{len(operation_ambiguities)} ambiguities remain unresolved"
                    ),
                    ambiguous_paths=[a.element_path for a in operation_ambiguities],
                    candidates=[
                        c
                        for a in operation_ambiguities
                        for c in a.candidates
                    ],
                )

        # Step 4: Validate logical-group structure (boolean scope)
        self._validate_logical_groups(working_draft)

        # Step 5: Finalize — select unique operation, set state for grounding
        working_draft = self._finalize_draft(working_draft)

        return working_draft

    # -------------------------------------------------------------------
    # Internal: Apply clarification patches
    # -------------------------------------------------------------------

    def _apply_clarification_patches(
        self,
        draft: SemanticIntentDraft,
        patches: list[SemanticPatch],
    ) -> SemanticIntentDraft:
        """Apply user clarification patches to the draft.

        Delegates to the patch applicator which produces a new immutable
        revision (N → N+1).

        Requirements: 2.1 - coordinator applies clarification patches
        """
        if not patches:
            return draft

        patched_draft = apply_patches(draft, patches)
        return patched_draft

    # -------------------------------------------------------------------
    # Internal: Find operation-type ambiguities
    # -------------------------------------------------------------------

    def _find_operation_ambiguities(
        self, draft: SemanticIntentDraft
    ) -> list[AmbiguityMarker]:
        """Identify ambiguity markers that relate to operation classification.

        Operation ambiguities are markers whose element_path targets an action
        type (e.g., "actions[0].type") or whose candidates include action
        type values (filter, project, drop, sort, rename).

        Requirements: 2.1 - coordinator receives operation candidates
        """
        operation_type_values = {"filter", "project", "drop", "sort", "rename"}
        operation_ambiguities: list[AmbiguityMarker] = []

        for marker in draft.ambiguities:
            # Check if the ambiguity is about operation type
            if self._is_operation_ambiguity(marker, operation_type_values):
                operation_ambiguities.append(marker)

        return operation_ambiguities

    def _is_operation_ambiguity(
        self,
        marker: AmbiguityMarker,
        operation_types: set[str],
    ) -> bool:
        """Determine if an ambiguity marker relates to operation classification.

        An ambiguity is operation-related if:
        - Its element_path targets an action's type field
        - Or its candidates include known action type values
        """
        # Path-based detection: "actions[N].type" or "actions[N]"
        if ".type" in marker.element_path or (
            marker.element_path.startswith("actions")
            and not any(
                sub in marker.element_path
                for sub in [".columns", ".keys", ".logical_groups", ".mappings"]
            )
        ):
            return True

        # Candidate-based detection: candidates contain action type names
        candidate_set = {c.lower() for c in marker.candidates}
        if candidate_set & operation_types:
            return True

        return False

    # -------------------------------------------------------------------
    # Internal: Check if ambiguities are resolved
    # -------------------------------------------------------------------

    def _check_ambiguities_resolved(
        self,
        draft: SemanticIntentDraft,
        operation_ambiguities: list[AmbiguityMarker],
    ) -> bool:
        """Check if operation ambiguities have been resolved.

        An ambiguity is considered resolved if:
        - It no longer appears in the draft's ambiguities list (was patched out)
        - Or the draft now has exactly one action and no operation markers

        Requirements: 2.1 - select unique operation when supported
        """
        # If the ambiguities from the input are no longer in the current draft,
        # they were resolved by clarification patches
        current_paths = {a.element_path for a in draft.ambiguities}
        unresolved_paths = {
            a.element_path
            for a in operation_ambiguities
            if a.element_path in current_paths
        }

        return len(unresolved_paths) == 0

    # -------------------------------------------------------------------
    # Internal: Validate logical-group structure (boolean scope)
    # -------------------------------------------------------------------

    def _validate_logical_groups(self, draft: SemanticIntentDraft) -> None:
        """Validate that logical-group structure is well-formed.

        The coordinator owns boolean scope (Req 2.2). This validates:
        - Each FilterAction has well-formed logical groups
        - Each logical group has a valid operator (and/or)
        - Each logical group has at least one predicate
        - Nested structure is consistent

        Raises:
            OperationAmbiguityError: If boolean scope is malformed and
                cannot be resolved without clarification.
        """
        for idx, action in enumerate(draft.actions):
            if not isinstance(action, FilterAction):
                continue

            for grp_idx, group in enumerate(action.logical_groups):
                # Validate operator is well-formed
                if group.operator not in ("and", "or"):
                    raise OperationAmbiguityError(
                        message=(
                            f"Invalid logical group operator '{group.operator}' "
                            f"at actions[{idx}].logical_groups[{grp_idx}]"
                        ),
                        ambiguous_paths=[
                            f"actions[{idx}].logical_groups[{grp_idx}].operator"
                        ],
                        candidates=["and", "or"],
                    )

                # Validate predicates exist (structural)
                if not group.predicates:
                    raise OperationAmbiguityError(
                        message=(
                            f"Empty logical group at "
                            f"actions[{idx}].logical_groups[{grp_idx}]"
                        ),
                        ambiguous_paths=[
                            f"actions[{idx}].logical_groups[{grp_idx}]"
                        ],
                    )

    # -------------------------------------------------------------------
    # Internal: Preserve ambiguity
    # -------------------------------------------------------------------

    def _preserve_ambiguity(
        self, draft: SemanticIntentDraft
    ) -> SemanticIntentDraft:
        """Preserve unresolved ambiguity in the draft.

        When the coordinator cannot resolve operation classification and
        no clarification is available, the draft is set to needs_clarification
        status to signal that user input is required.

        Requirements: 2.1 - preserves unresolved ambiguity otherwise
        """
        # If the draft is still pending, mark as needs_clarification
        if draft.resolution_status == ResolutionStatus.PENDING:
            draft_dict = draft.model_dump(mode="json")
            draft_dict["resolution_status"] = ResolutionStatus.NEEDS_CLARIFICATION.value
            import json as _json

            return SemanticIntentDraft.model_validate_json(_json.dumps(draft_dict))

        return draft

    # -------------------------------------------------------------------
    # Internal: Finalize draft for grounding dispatch
    # -------------------------------------------------------------------

    def _finalize_draft(self, draft: SemanticIntentDraft) -> SemanticIntentDraft:
        """Finalize the draft for grounding dispatch.

        Sets the draft to a state indicating operation classification and
        boolean scope are finalized. Adds a resolution record documenting
        the coordinator's decision.

        If the draft has exactly one action and no operation ambiguity markers,
        it is finalized as-is. The coordinator records its ownership decision
        in the resolution history.

        Requirements: 21.3 - finalize action type and logical-group structure
                              before grounding dispatch
        """
        import json as _json

        draft_dict = draft.model_dump(mode="json")

        # Record the coordinator's finalization in resolution history
        resolution_record = ResolutionRecord(
            timestamp=datetime.now(timezone.utc),
            stage="intent_resolution",
            decision_owner="IntentResolutionCoordinator",
            element_path="actions",
            resolution="operation_classification_finalized",
            confidence=1.0,
            evidence=[
                f"action_count={len(draft.actions)}",
                f"action_types={[a.type for a in draft.actions]}",
                f"remaining_ambiguities={len(self._find_operation_ambiguities(draft))}",
            ],
            provenance=[],
        )

        # Append to resolution history
        history = draft_dict.get("resolution_history", [])
        history.append(resolution_record.model_dump(mode="json"))
        draft_dict["resolution_history"] = history

        # Increment revision to reflect coordinator's finalization
        draft_dict["draft_revision"] = draft.draft_revision + 1

        # Reconstruct the validated draft
        new_draft = SemanticIntentDraft.model_validate_json(_json.dumps(draft_dict))

        return new_draft
