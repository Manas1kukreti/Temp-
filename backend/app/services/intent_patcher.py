"""
IntentPatcher — applies selective modifications to the CanonicalIntent.

Patches only the unresolved fields targeted by user clarification answers,
preserving all other fields. Handles:
- Direct candidate_option selection → resolved_column with confidence=1.0
- "none_of_these" + free_text → raw_reference update + column re-resolution
- Intent versioning (increment revision, compute SHA-256 hash, set parent_intent_id)
- CanonicalIntentRevision persistence for audit trail
- Re-grounding and validation after patching
- Conflict detection

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.1, 8.2, 8.3
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from app.schemas.clarification import ClarificationAnswer
from app.services.clarification_questions import UnresolvedField

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for IntentPackage (to avoid tight coupling to agent-framework)
# ---------------------------------------------------------------------------


class IntentPackageLike(Protocol):
    """Protocol matching the IntentPackage interface used by the patcher."""

    unresolved_fields: list[str]

    def patch_column(
        self,
        requested_field: str,
        new_column: str,
        reason: str,
    ) -> "IntentPackageLike": ...


# ---------------------------------------------------------------------------
# PatchResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class PatchResult:
    """Result of applying an intent patch.

    Attributes:
        intent: The patched CanonicalIntent (dict representation).
        remaining_unresolved_fields: Unresolved fields still present after re-grounding.
        conflict_flag: True if the patch introduced a conflict (e.g., same column
            referenced in contradictory operations).
        new_revision: The new intent_revision number.
        new_hash: The SHA-256 hash of the patched intent.
    """

    intent: dict[str, Any]
    remaining_unresolved_fields: list[UnresolvedField]
    conflict_flag: bool
    new_revision: int
    new_hash: str


# ---------------------------------------------------------------------------
# IntentPatcher
# ---------------------------------------------------------------------------


class IntentPatcher:
    """Applies selective modifications to a CanonicalIntent based on clarification answers.

    The patcher:
    1. Applies each answer to the targeted unresolved field in the intent
    2. Patches the IntentPackage's resolved columns accordingly
    3. Increments intent_revision, computes new intent_hash, sets parent_intent_id
    4. Persists a CanonicalIntentRevision record with the previous version state
    5. Re-runs grounding and validation (skips extraction/normalization)
    6. Detects conflicts and sets conflict_flag in PatchResult
    """

    async def apply_patch(
        self,
        intent: dict[str, Any],
        intent_package: Any,
        answers: list[ClarificationAnswer],
        *,
        questions: list[Any] | None = None,
        db: Any | None = None,
        submission: Any | None = None,
    ) -> PatchResult:
        """Patch unresolved fields based on user clarification answers.

        Args:
            intent: The current CanonicalIntent as a dict.
            intent_package: The IntentPackage schema-resolution artifact.
            answers: User-provided clarification answers.
            questions: The ClarificationQuestion objects for mapping answers to intent_paths.
            db: Optional async database session for persisting revisions.
            submission: Optional Submission model for updating intent metadata.

        Returns:
            PatchResult with the patched intent, remaining unresolved fields,
            conflict flag, new revision number, and new hash.
        """
        # Save previous state for revision persistence
        previous_intent = copy.deepcopy(intent)
        previous_intent_id = intent.get("intent_id", str(uuid.uuid4()))
        previous_revision = intent.get("intent_revision", 1)

        # Build question lookup: question_id → question object
        question_map: dict[str, Any] = {}
        if questions:
            for q in questions:
                q_id = str(q.id) if hasattr(q, "id") else str(q.get("id", ""))
                question_map[q_id] = q

        # Apply each answer to the intent
        patched_intent = copy.deepcopy(intent)
        current_package = intent_package

        for answer in answers:
            question = question_map.get(str(answer.question_id))
            if not question:
                continue

            intent_path = _get_intent_path(question)
            if not intent_path:
                continue

            if answer.selected_option and answer.selected_option != "none_of_these":
                # Direct selection: set resolved_column with confidence=1.0
                _apply_selected_option(patched_intent, intent_path, answer.selected_option)
                current_package = _patch_package_column(
                    current_package, intent_path, answer.selected_option
                )
            elif answer.selected_option == "none_of_these" and answer.free_text:
                # Free-text: update raw_reference and re-run column resolution
                _apply_free_text(patched_intent, intent_path, answer.free_text.strip())
                # Re-resolve: attempt column resolution against the free_text
                resolved = _resolve_free_text_column(
                    current_package, intent_path, answer.free_text.strip()
                )
                if resolved:
                    _apply_selected_option(patched_intent, intent_path, resolved)
                    current_package = _patch_package_column(
                        current_package, intent_path, resolved
                    )

        # Increment revision
        new_revision = previous_revision + 1
        patched_intent["intent_revision"] = new_revision

        # Set parent_intent_id
        patched_intent["parent_intent_id"] = previous_intent_id

        # Generate new intent_id for this revision
        new_intent_id = str(uuid.uuid4())
        patched_intent["intent_id"] = new_intent_id

        # Compute new intent_hash (SHA-256)
        new_hash = _compute_intent_hash(patched_intent)
        patched_intent["intent_hash"] = new_hash

        # Persist CanonicalIntentRevision with previous version's full state
        if db and submission:
            await _persist_revision(
                db=db,
                submission=submission,
                previous_intent=previous_intent,
                previous_intent_id=previous_intent_id,
                previous_revision=previous_revision,
                new_intent_id=new_intent_id,
                new_revision=new_revision,
                new_hash=new_hash,
            )

        # Re-run grounding and validation (skip extraction/normalization)
        remaining_unresolved = _re_ground_and_validate(patched_intent, current_package)

        # Detect conflicts
        conflict_flag = _detect_conflicts(patched_intent)

        return PatchResult(
            intent=patched_intent,
            remaining_unresolved_fields=remaining_unresolved,
            conflict_flag=conflict_flag,
            new_revision=new_revision,
            new_hash=new_hash,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_intent_path(question: Any) -> str:
    """Extract intent_path from a question object or dict."""
    if hasattr(question, "intent_path"):
        return question.intent_path
    if isinstance(question, dict):
        return question.get("intent_path", "")
    return ""


def _apply_selected_option(intent: dict[str, Any], intent_path: str, selected: str) -> None:
    """Apply a selected candidate option to the intent at the given path.

    Sets resolved_column to the selection, confidence=1.0,
    resolution_method="user_clarification".
    """
    target = _navigate_intent_path(intent, intent_path)
    if target is not None and isinstance(target, dict):
        target["resolved_column"] = selected
        target["confidence"] = 1.0
        target["resolution_method"] = "user_clarification"
        # Clear candidate columns since user has resolved
        if "candidate_columns" in target:
            target["candidate_columns"] = [selected]
    else:
        # Try to find the parent and set the field reference
        parent, key = _navigate_to_parent(intent, intent_path)
        if parent is not None and isinstance(parent, dict):
            if key and key in parent and isinstance(parent[key], dict):
                parent[key]["resolved_column"] = selected
                parent[key]["confidence"] = 1.0
                parent[key]["resolution_method"] = "user_clarification"


def _apply_free_text(intent: dict[str, Any], intent_path: str, free_text: str) -> None:
    """Apply free_text to the intent, updating raw_reference at the given path."""
    target = _navigate_intent_path(intent, intent_path)
    if target is not None and isinstance(target, dict):
        target["raw_reference"] = free_text
        # Clear previous resolution since we're re-resolving
        target["resolved_column"] = None
        target["resolution_method"] = None
        target["confidence"] = 0.0
    else:
        parent, key = _navigate_to_parent(intent, intent_path)
        if parent is not None and isinstance(parent, dict):
            if key and key in parent and isinstance(parent[key], dict):
                parent[key]["raw_reference"] = free_text
                parent[key]["resolved_column"] = None
                parent[key]["resolution_method"] = None


def _navigate_intent_path(intent: dict[str, Any], path: str) -> Any:
    """Navigate a JSON-path style string to find the target in the intent.

    Supports paths like: "actions[1].conditions[0].field" or
    "actions[1].conditions[0].field.raw_reference"
    """
    if not path:
        return None

    current: Any = intent
    segments = _parse_path_segments(path)

    for segment in segments:
        if current is None:
            return None

        if isinstance(segment, int):
            if isinstance(current, list) and 0 <= segment < len(current):
                current = current[segment]
            else:
                return None
        elif isinstance(segment, str):
            if isinstance(current, dict) and segment in current:
                current = current[segment]
            else:
                return None
        else:
            return None

    return current


def _navigate_to_parent(intent: dict[str, Any], path: str) -> tuple[Any, str | None]:
    """Navigate to the parent object and return (parent, last_key)."""
    if not path:
        return None, None

    segments = _parse_path_segments(path)
    if len(segments) < 2:
        return intent, segments[0] if segments else None

    current: Any = intent
    for segment in segments[:-1]:
        if current is None:
            return None, None
        if isinstance(segment, int):
            if isinstance(current, list) and 0 <= segment < len(current):
                current = current[segment]
            else:
                return None, None
        elif isinstance(segment, str):
            if isinstance(current, dict) and segment in current:
                current = current[segment]
            else:
                return None, None

    last = segments[-1]
    return current, last if isinstance(last, str) else None


def _parse_path_segments(path: str) -> list[str | int]:
    """Parse a JSON-path-like string into segments.

    Examples:
        "actions[1].conditions[0].field" → ["actions", 1, "conditions", 0, "field"]
        "actions[0].requested_fields[2]" → ["actions", 0, "requested_fields", 2]
    """
    segments: list[str | int] = []
    # Split by '.' but handle bracket notation
    parts = re.split(r"\.", path)
    for part in parts:
        if not part:
            continue
        # Handle bracket indexing: "actions[1]" → "actions", 1
        bracket_match = re.match(r"^(\w+)\[(\d+)\]$", part)
        if bracket_match:
            segments.append(bracket_match.group(1))
            segments.append(int(bracket_match.group(2)))
        elif re.match(r"^\[\d+\]$", part):
            segments.append(int(part[1:-1]))
        else:
            segments.append(part)
    return segments


def _patch_package_column(
    intent_package: Any, intent_path: str, resolved_column: str
) -> Any:
    """Patch the IntentPackage resolved columns for the given field.

    Uses the IntentPackage.patch_column() method if available,
    otherwise returns the package as-is.
    """
    # Extract the requested_field identifier from the intent_path
    requested_field = _intent_path_to_requested_field(intent_path)

    if hasattr(intent_package, "patch_column"):
        try:
            return intent_package.patch_column(
                requested_field, resolved_column, "user_clarification"
            )
        except Exception as e:
            logger.warning("Failed to patch IntentPackage column: %s", e)
            return intent_package

    return intent_package


def _resolve_free_text_column(
    intent_package: Any, intent_path: str, free_text: str
) -> str | None:
    """Attempt to resolve free_text to a column using the intent package.

    Checks if the free_text matches any known column in the package's
    resolved columns or semantic profiles.
    """
    # Try exact match against resolved columns
    if hasattr(intent_package, "resolved_columns"):
        for rc in intent_package.resolved_columns:
            col_name = rc.resolved_column if hasattr(rc, "resolved_column") else str(rc)
            if col_name.lower() == free_text.lower():
                return col_name

    # Try matching against semantic profiles
    if hasattr(intent_package, "semantic_profiles"):
        for profile in intent_package.semantic_profiles:
            col_name = profile.column_name if hasattr(profile, "column_name") else str(profile)
            if col_name.lower() == free_text.lower():
                return col_name

    # No exact match found — return None (field remains unresolved)
    return None


def _intent_path_to_requested_field(intent_path: str) -> str:
    """Convert an intent_path to a requested_field identifier.

    For paths like "actions[1].conditions[0].field.raw_reference",
    returns the path up to the field reference container.
    """
    # Strip trailing .raw_reference or .resolved_column suffixes
    cleaned = re.sub(r"\.(raw_reference|resolved_column|resolution_method|confidence)$", "", intent_path)
    return cleaned


def _compute_intent_hash(intent: dict[str, Any]) -> str:
    """Compute SHA-256 hash of the intent, excluding mutable metadata fields."""
    payload = dict(intent)
    # Exclude fields that change with each revision
    for key in (
        "intent_id",
        "intent_revision",
        "intent_hash",
        "parent_intent_id",
        "created_at",
        "grounded_at",
    ):
        payload.pop(key, None)

    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def _persist_revision(
    *,
    db: Any,
    submission: Any,
    previous_intent: dict[str, Any],
    previous_intent_id: str,
    previous_revision: int,
    new_intent_id: str,
    new_revision: int,
    new_hash: str,
) -> None:
    """Persist a CanonicalIntentRevision record with the previous version state."""
    from app.models import CanonicalIntentRevision

    previous_hash = previous_intent.get("intent_hash", _compute_intent_hash(previous_intent))

    revision_record = CanonicalIntentRevision(
        submission_id=submission.id,
        intent_id=uuid.UUID(previous_intent_id) if isinstance(previous_intent_id, str) else previous_intent_id,
        intent_revision=previous_revision,
        intent_hash=previous_hash,
        parent_intent_id=uuid.UUID(previous_intent.get("parent_intent_id")) if previous_intent.get("parent_intent_id") else None,
        canonical_intent=previous_intent,
        original_instruction=previous_intent.get("original_prompt", ""),
        grounded_at=datetime.now(UTC),
        capability_version=previous_intent.get("capability_version"),
        extractor_version=previous_intent.get("extractor_version"),
        normalizer_version=previous_intent.get("normalizer_version"),
        grounding_version=previous_intent.get("grounding_version"),
    )
    db.add(revision_record)
    await db.flush()

    # Update submission's intent metadata to reflect new version
    submission.intent_id = uuid.UUID(new_intent_id)
    submission.intent_revision = new_revision
    submission.intent_hash = new_hash
    submission.parent_intent_id = uuid.UUID(previous_intent_id) if isinstance(previous_intent_id, str) else previous_intent_id
    await db.flush()


def _re_ground_and_validate(
    intent: dict[str, Any], intent_package: Any
) -> list[UnresolvedField]:
    """Re-run grounding and validation on the patched intent.

    Skips extraction and normalization (the intent structure is already built).
    Only checks for remaining unresolved column references.
    """
    from app.models.clarification import ReasonCode

    remaining: list[UnresolvedField] = []

    actions = intent.get("actions", [])
    for action_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue

        kind = action.get("kind", "")

        # Check requested_fields (project_columns, drop_columns)
        for field_idx, field_ref in enumerate(action.get("requested_fields", [])):
            if not isinstance(field_ref, dict):
                continue
            if not field_ref.get("resolved_column") and not field_ref.get("resolved_columns"):
                intent_path = f"actions[{action_idx}].requested_fields[{field_idx}]"
                remaining.append(
                    UnresolvedField(
                        intent_path=intent_path,
                        reason_code=_determine_reason_code(field_ref),
                        raw_reference=field_ref.get("raw_reference", ""),
                        grounding_candidates=_extract_candidates(field_ref),
                    )
                )

        # Check filter conditions
        for cond_idx, condition in enumerate(action.get("conditions", [])):
            if not isinstance(condition, dict):
                continue
            field_ref = condition.get("field", {})
            if isinstance(field_ref, dict) and not field_ref.get("resolved_column"):
                intent_path = f"actions[{action_idx}].conditions[{cond_idx}].field"
                remaining.append(
                    UnresolvedField(
                        intent_path=intent_path,
                        reason_code=_determine_reason_code(field_ref),
                        raw_reference=field_ref.get("raw_reference", ""),
                        grounding_candidates=_extract_candidates(field_ref),
                    )
                )

        # Check sort_keys
        for key_idx, sort_key in enumerate(action.get("sort_keys", [])):
            if not isinstance(sort_key, dict):
                continue
            col_ref = sort_key.get("column", {})
            if isinstance(col_ref, dict) and not col_ref.get("resolved_column"):
                intent_path = f"actions[{action_idx}].sort_keys[{key_idx}].column"
                remaining.append(
                    UnresolvedField(
                        intent_path=intent_path,
                        reason_code=_determine_reason_code(col_ref),
                        raw_reference=col_ref.get("raw_reference", ""),
                        grounding_candidates=_extract_candidates(col_ref),
                    )
                )

        # Check rename mapping
        for map_idx, mapping in enumerate(action.get("mapping", [])):
            if not isinstance(mapping, dict):
                continue
            source_ref = mapping.get("source", {})
            if isinstance(source_ref, dict) and not source_ref.get("resolved_column"):
                intent_path = f"actions[{action_idx}].mapping[{map_idx}].source"
                remaining.append(
                    UnresolvedField(
                        intent_path=intent_path,
                        reason_code=_determine_reason_code(source_ref),
                        raw_reference=source_ref.get("raw_reference", ""),
                        grounding_candidates=_extract_candidates(source_ref),
                    )
                )

        # Check visualize fields
        for viz_idx, viz_field in enumerate(action.get("fields", [])):
            if not isinstance(viz_field, dict):
                continue
            if action.get("kind") == "visualize" and not viz_field.get("resolved_column"):
                intent_path = f"actions[{action_idx}].fields[{viz_idx}]"
                remaining.append(
                    UnresolvedField(
                        intent_path=intent_path,
                        reason_code=_determine_reason_code(viz_field),
                        raw_reference=viz_field.get("raw_reference", ""),
                        grounding_candidates=_extract_candidates(viz_field),
                    )
                )

    # Update intent resolution status based on remaining fields
    if remaining:
        intent["resolution_status"] = "needs_clarification"
    else:
        intent["resolution_status"] = "resolved"

    return remaining


def _determine_reason_code(field_ref: dict[str, Any]) -> "ReasonCode":
    """Determine the reason code for an unresolved field reference."""
    from app.models.clarification import ReasonCode

    candidates = field_ref.get("candidate_columns", [])

    if len(candidates) > 1:
        return ReasonCode.MULTIPLE_COLUMN_MATCHES
    elif len(candidates) == 0:
        return ReasonCode.MISSING_COLUMN
    else:
        # Single candidate with low confidence
        return ReasonCode.LOW_CONFIDENCE_SCORE


def _extract_candidates(field_ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract grounding candidates from a field reference."""
    candidates = field_ref.get("candidate_columns", [])
    return [
        {"column_name": c, "confidence": 0.5}
        for c in candidates
        if isinstance(c, str)
    ]


def _detect_conflicts(intent: dict[str, Any]) -> bool:
    """Detect if the patched intent has conflicts.

    A conflict occurs when the same column is referenced in contradictory
    operations (e.g., both kept and dropped, or used in conflicting filters).
    """
    actions = intent.get("actions", [])

    # Collect columns that are kept/projected
    projected_columns: set[str] = set()
    # Collect columns that are dropped
    dropped_columns: set[str] = set()

    for action in actions:
        if not isinstance(action, dict):
            continue

        kind = action.get("kind", "")

        if kind == "project_columns":
            for field_ref in action.get("requested_fields", []):
                if isinstance(field_ref, dict) and field_ref.get("resolved_column"):
                    projected_columns.add(field_ref["resolved_column"])

        elif kind == "drop_columns":
            for field_ref in action.get("requested_fields", []):
                if isinstance(field_ref, dict) and field_ref.get("resolved_column"):
                    dropped_columns.add(field_ref["resolved_column"])

    # Conflict: column both projected and dropped
    if projected_columns and dropped_columns:
        conflict = projected_columns & dropped_columns
        if conflict:
            logger.warning("Conflict detected: columns both projected and dropped: %s", conflict)
            return True

    return False
