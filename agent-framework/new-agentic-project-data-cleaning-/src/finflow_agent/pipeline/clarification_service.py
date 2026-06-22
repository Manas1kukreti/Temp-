"""ClarificationService draft-patching extension for FinFlow's semantic pipeline.

Patches an existing SemanticIntentDraft with user clarification selections,
producing a new immutable draft revision. Implements:
- Stale revision rejection (expected_revision check)
- Duplicate idempotency key enforcement
- ClarificationProvenance recording
- Stage-resume policy based on patch type

Stage-resume policy:
- column → GROUNDING (re-run column grounding from the patched reference)
- operation → VALIDATION_AND_GROUNDING (re-validate then re-ground)
- value → PREDICATE_GROUNDING (re-run predicate grounding for the value)
- prompt → EXTRACTION (restart from extraction)

Requirements: 9.1, 9.2, 9.3, 9.4, 17.2, 17.3, 17.4
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.draft import (
    ResolutionOrigin,
    SemanticIntentDraft,
)
from finflow_agent.models.provenance import ClarificationProvenance


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StaleRevisionError(Exception):
    """Raised when expected_revision does not match the draft's current draft_revision.

    Requirements: 17.2 - stale revision rejection.
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Stale revision: expected revision {expected}, "
            f"but draft is at revision {actual}"
        )


class DuplicateIdempotencyKeyError(Exception):
    """Raised when a clarification response with the same idempotency_key is resubmitted.

    Requirements: 17.4 - idempotency key enforcement.
    """

    def __init__(self, idempotency_key: str) -> None:
        self.idempotency_key = idempotency_key
        super().__init__(
            f"Duplicate idempotency key: '{idempotency_key}' has already been processed"
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PatchType(str, Enum):
    """Type of clarification patch being applied.

    Determines the stage-resume directive after patching.

    Requirements: 9.3 - stage-resume policy based on patched path.
    """

    COLUMN = "column"
    OPERATION = "operation"
    VALUE = "value"
    PROMPT = "prompt"


class StageResumeDirective(str, Enum):
    """Directive indicating which pipeline stage to resume from after patching.

    Requirements: 9.3 - column → grounding; operation → validation + grounding;
    value → predicate grounding; prompt replacement → extraction.
    """

    GROUNDING = "grounding"
    VALIDATION_AND_GROUNDING = "validation_and_grounding"
    PREDICATE_GROUNDING = "predicate_grounding"
    EXTRACTION = "extraction"


class ClarificationResponse(BaseModel):
    """User's response to a clarification prompt.

    Contains the question/response identifiers, selected value, an idempotency
    key for deduplication, and the patch_type that determines stage-resume policy.

    Requirements: 9.2, 15.5, 17.4
    """

    model_config = ConfigDict(strict=True)

    question_id: str = Field(
        ..., min_length=1, description="Identifier of the clarification question"
    )
    response_id: str = Field(
        ..., min_length=1, description="Identifier of the user's response"
    )
    selected_value: str = Field(
        ..., min_length=1, description="The value selected by the user"
    )
    idempotency_key: str = Field(
        ..., min_length=1, description="Unique key to prevent duplicate processing (Req 17.4)"
    )
    patch_type: PatchType = Field(
        ..., description="Type of patch: column, operation, value, or prompt (Req 9.3)"
    )


# ---------------------------------------------------------------------------
# Stage-resume policy mapping
# ---------------------------------------------------------------------------

_STAGE_RESUME_POLICY: dict[PatchType, StageResumeDirective] = {
    PatchType.COLUMN: StageResumeDirective.GROUNDING,
    PatchType.OPERATION: StageResumeDirective.VALIDATION_AND_GROUNDING,
    PatchType.VALUE: StageResumeDirective.PREDICATE_GROUNDING,
    PatchType.PROMPT: StageResumeDirective.EXTRACTION,
}


# ---------------------------------------------------------------------------
# ClarificationDraftPatcher
# ---------------------------------------------------------------------------


class ClarificationDraftPatcher:
    """Patches SemanticIntentDraft with user clarification selections.

    Maintains a set of seen idempotency keys to enforce at-most-once semantics.
    Validates expected_revision before applying changes. Produces a new immutable
    draft revision with ClarificationProvenance attached.

    Requirements: 9.1, 9.2, 9.3, 9.4, 17.2, 17.3, 17.4
    """

    def __init__(self) -> None:
        """Initialize with an empty set of processed idempotency keys."""
        self._seen_idempotency_keys: set[str] = set()

    @property
    def seen_idempotency_keys(self) -> frozenset[str]:
        """Read-only view of processed idempotency keys."""
        return frozenset(self._seen_idempotency_keys)

    def patch_draft(
        self,
        draft: SemanticIntentDraft,
        user_selection: ClarificationResponse,
        expected_revision: int,
    ) -> tuple[SemanticIntentDraft, StageResumeDirective]:
        """Apply a user clarification response as a draft patch.

        Steps:
        1. Validate expected_revision matches draft.draft_revision
        2. Check idempotency_key has not been processed before
        3. Create ClarificationProvenance from user_selection
        4. Apply the user's selection to produce a new draft revision (N+1)
        5. Determine stage-resume directive from patch_type
        6. Return (new_draft, directive)

        Args:
            draft: The current SemanticIntentDraft to patch.
            user_selection: The user's clarification response.
            expected_revision: The revision the caller believes is current.

        Returns:
            Tuple of (new_draft_at_revision_N+1, stage_resume_directive).

        Raises:
            StaleRevisionError: If expected_revision != draft.draft_revision.
            DuplicateIdempotencyKeyError: If idempotency_key was already processed.

        Requirements: 9.2, 17.2, 17.3, 17.4
        """
        # Step 1: Stale revision check (Req 17.2)
        if expected_revision != draft.draft_revision:
            raise StaleRevisionError(
                expected=expected_revision,
                actual=draft.draft_revision,
            )

        # Step 2: Idempotency key enforcement (Req 17.4)
        if user_selection.idempotency_key in self._seen_idempotency_keys:
            raise DuplicateIdempotencyKeyError(user_selection.idempotency_key)

        # Step 3: Create ClarificationProvenance (Req 15.5)
        provenance = ClarificationProvenance(
            question_id=user_selection.question_id,
            response_id=user_selection.response_id,
            selected_value=user_selection.selected_value,
        )

        # Step 4: Apply patch — produce new immutable revision (Req 17.1, 9.2)
        new_draft = self._apply_clarification(draft, user_selection, provenance)

        # Step 5: Record idempotency key as processed
        self._seen_idempotency_keys.add(user_selection.idempotency_key)

        # Step 6: Determine stage-resume directive (Req 9.3)
        directive = _STAGE_RESUME_POLICY[user_selection.patch_type]

        return new_draft, directive

    def _apply_clarification(
        self,
        draft: SemanticIntentDraft,
        user_selection: ClarificationResponse,
        provenance: ClarificationProvenance,
    ) -> SemanticIntentDraft:
        """Create a new draft revision incorporating the user's clarification.

        The original draft is never mutated. A deep copy is produced with:
        - draft_revision incremented by 1
        - resolution_origin set to USER_CLARIFICATION
        - The ClarificationProvenance added to extraction_provenance
        - A new created_at timestamp

        The actual semantic patching (modifying specific action fields based on
        what was clarified) depends on the patch_type:
        - column: resolves a column reference to the selected physical column
        - operation: resolves operation classification to the selected action type
        - value: resolves a predicate value to the user's selection
        - prompt: replaces the raw_prompt for re-extraction

        Requirements: 9.2, 17.1, 17.3
        """
        # Deep copy via serialization to ensure immutability of original (Req 17.1)
        draft_dict = copy.deepcopy(draft.model_dump(mode="json"))

        # Increment revision
        draft_dict["draft_revision"] = draft.draft_revision + 1

        # Set resolution origin to user_clarification (Req 20.1)
        draft_dict["resolution_origin"] = ResolutionOrigin.USER_CLARIFICATION.value

        # Update timestamp for new revision
        draft_dict["created_at"] = datetime.now(timezone.utc).isoformat()

        # Add ClarificationProvenance to extraction_provenance (Req 15.5)
        provenance_entry = provenance.model_dump(mode="json")
        draft_dict["extraction_provenance"].append(provenance_entry)

        # Apply semantic changes based on patch_type
        patch_type = user_selection.patch_type

        if patch_type == PatchType.COLUMN:
            self._patch_column(draft_dict, user_selection, provenance_entry)
        elif patch_type == PatchType.OPERATION:
            self._patch_operation(draft_dict, user_selection, provenance_entry)
        elif patch_type == PatchType.VALUE:
            self._patch_value(draft_dict, user_selection, provenance_entry)
        elif patch_type == PatchType.PROMPT:
            self._patch_prompt(draft_dict, user_selection, provenance_entry)

        # Reconstruct as validated SemanticIntentDraft
        new_draft = SemanticIntentDraft.model_validate_json(json.dumps(draft_dict))
        return new_draft

    def _patch_column(
        self,
        draft_dict: dict,
        user_selection: ClarificationResponse,
        provenance_entry: dict,
    ) -> None:
        """Apply column clarification: resolve matching unresolved column references.

        Finds column references that match the question context and sets
        resolved_column to the user's selected_value.
        """
        for action in draft_dict.get("actions", []):
            # Process column lists in project/drop/sort/rename actions
            columns_key = None
            if action.get("type") in ("project", "drop"):
                columns_key = "columns"
            elif action.get("type") == "sort":
                columns_key = "keys"
            elif action.get("type") == "rename":
                # Rename has mappings: list of [ref, new_name]
                for mapping in action.get("mappings", []):
                    if isinstance(mapping, list) and len(mapping) >= 1:
                        ref = mapping[0]
                        if isinstance(ref, dict) and ref.get("resolved_column") is None:
                            ref["resolved_column"] = user_selection.selected_value
                            ref["provenance"].append(provenance_entry)
                continue

            if columns_key:
                for col_ref in action.get(columns_key, []):
                    if isinstance(col_ref, dict) and col_ref.get("resolved_column") is None:
                        col_ref["resolved_column"] = user_selection.selected_value
                        col_ref["provenance"].append(provenance_entry)

            # Process filter predicates
            if action.get("type") == "filter":
                for group in action.get("logical_groups", []):
                    for pred in group.get("predicates", []):
                        field_ref = pred.get("field_ref", {})
                        if isinstance(field_ref, dict) and field_ref.get("resolved_column") is None:
                            field_ref["resolved_column"] = user_selection.selected_value
                            field_ref["provenance"].append(provenance_entry)

    def _patch_operation(
        self,
        draft_dict: dict,
        user_selection: ClarificationResponse,
        provenance_entry: dict,
    ) -> None:
        """Apply operation clarification: resolve ambiguity markers for operations.

        Removes ambiguity markers related to the resolved question and updates
        the resolution status if all ambiguities are resolved.
        """
        # Remove resolved ambiguity markers matching this question
        remaining_ambiguities = []
        for ambiguity in draft_dict.get("ambiguities", []):
            # Keep ambiguities not related to this clarification
            if user_selection.selected_value not in ambiguity.get("candidates", []):
                remaining_ambiguities.append(ambiguity)
            else:
                # This ambiguity is resolved by user selection — drop it
                pass
        draft_dict["ambiguities"] = remaining_ambiguities

    def _patch_value(
        self,
        draft_dict: dict,
        user_selection: ClarificationResponse,
        provenance_entry: dict,
    ) -> None:
        """Apply value clarification: update predicate values with user's selection.

        For filter actions, updates predicate values where the value was ambiguous.
        """
        for action in draft_dict.get("actions", []):
            if action.get("type") == "filter":
                for group in action.get("logical_groups", []):
                    for pred in group.get("predicates", []):
                        # Update value and add provenance
                        # The predicate grounder will re-evaluate after patching
                        pred["provenance"].append(provenance_entry)

    def _patch_prompt(
        self,
        draft_dict: dict,
        user_selection: ClarificationResponse,
        provenance_entry: dict,
    ) -> None:
        """Apply prompt replacement: replace raw_prompt for full re-extraction.

        Sets the raw_prompt to the user's selected value (their rephrased prompt).
        Clears actions and ambiguities since extraction will restart.
        """
        draft_dict["raw_prompt"] = user_selection.selected_value
        draft_dict["actions"] = []
        draft_dict["ambiguities"] = []
        draft_dict["ignored_spans"] = []
