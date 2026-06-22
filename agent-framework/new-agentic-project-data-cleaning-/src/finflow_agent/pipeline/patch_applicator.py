"""Deterministic patch application logic for SemanticIntentDraft.

Applies SemanticPatch operations against the current draft to produce a new
immutable revision (N → N+1). The original draft is never mutated.

Path resolution uses a simplified JSON-path convention:
- "actions[0].columns[1].resolved_column" — navigate into draft structure
- "ambiguities[0]" — target ambiguity markers
- "actions" — target the actions list itself

Operations:
- ADD: insert value at path (for list items, append)
- REPLACE: set value at path to new value
- REMOVE: delete element at path (for list items, remove by index)

Requirements: 4.4, 15.4, 17.1
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from finflow_agent.models.draft import SemanticIntentDraft
from finflow_agent.models.patches import PatchOp, SemanticPatch


class PatchApplicationError(Exception):
    """Raised when a patch cannot be applied due to invalid path or operation failure."""

    def __init__(self, message: str, patch: SemanticPatch | None = None) -> None:
        self.patch = patch
        super().__init__(message)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Matches segments like "actions", "columns[0]", "logical_groups[2]"
_SEGMENT_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\[(\d+)\])?$")


def _parse_path(path: str) -> list[str | int]:
    """Parse a simplified JSON-path string into a list of keys/indices.

    Examples:
        "actions[0].columns[1].resolved_column"
        → ["actions", 0, "columns", 1, "resolved_column"]

        "ambiguities[0]"
        → ["ambiguities", 0]

        "actions"
        → ["actions"]
    """
    if not path:
        raise PatchApplicationError(f"Empty patch path: '{path}'")

    segments = path.split(".")
    tokens: list[str | int] = []

    for segment in segments:
        match = _SEGMENT_PATTERN.match(segment)
        if not match:
            raise PatchApplicationError(
                f"Invalid path segment '{segment}' in path '{path}'"
            )
        tokens.append(match.group(1))
        if match.group(2) is not None:
            tokens.append(int(match.group(2)))

    return tokens


def _resolve_target(obj: Any, tokens: list[str | int]) -> tuple[Any, str | int]:
    """Walk the token list, returning (parent_container, final_key).

    Navigates through dicts (Pydantic model dicts) and lists using the parsed
    tokens. Returns the immediate parent and the final key/index so the caller
    can apply add/replace/remove.
    """
    if not tokens:
        raise PatchApplicationError("Cannot resolve empty token path")

    current = obj
    for i, token in enumerate(tokens[:-1]):
        try:
            if isinstance(current, dict):
                current = current[token]
            elif isinstance(current, list):
                if not isinstance(token, int):
                    raise PatchApplicationError(
                        f"Expected integer index for list access, got '{token}'"
                    )
                current = current[token]
            else:
                raise PatchApplicationError(
                    f"Cannot navigate into {type(current).__name__} with token '{token}'"
                )
        except (KeyError, IndexError, TypeError) as exc:
            path_so_far = tokens[: i + 1]
            raise PatchApplicationError(
                f"Path resolution failed at {path_so_far!r}: {exc}"
            ) from exc

    return current, tokens[-1]


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


def _apply_add(parent: Any, key: str | int, value: Any) -> None:
    """ADD operation: insert value at path. For lists, append."""
    if isinstance(parent, list):
        if isinstance(key, int):
            # Insert at index position; if out of range, append
            if key >= len(parent):
                parent.append(value)
            else:
                parent.insert(key, value)
        else:
            raise PatchApplicationError(
                f"Cannot use string key '{key}' to add to a list"
            )
    elif isinstance(parent, dict):
        # For dicts, set the key (add new field or append to existing list field)
        if isinstance(key, str) and key in parent and isinstance(parent[key], list):
            parent[key].append(value)
        else:
            parent[key] = value
    else:
        raise PatchApplicationError(
            f"Cannot apply ADD to {type(parent).__name__} at key '{key}'"
        )


def _apply_replace(parent: Any, key: str | int, value: Any) -> None:
    """REPLACE operation: set value at path to new value."""
    if isinstance(parent, list):
        if not isinstance(key, int):
            raise PatchApplicationError(
                f"Cannot use string key '{key}' to replace in a list"
            )
        if key < 0 or key >= len(parent):
            raise PatchApplicationError(
                f"List index {key} out of range (length {len(parent)})"
            )
        parent[key] = value
    elif isinstance(parent, dict):
        if key not in parent:
            raise PatchApplicationError(
                f"Cannot REPLACE non-existent key '{key}' in dict"
            )
        parent[key] = value
    else:
        raise PatchApplicationError(
            f"Cannot apply REPLACE to {type(parent).__name__} at key '{key}'"
        )


def _apply_remove(parent: Any, key: str | int) -> None:
    """REMOVE operation: delete element at path. For lists, remove by index."""
    if isinstance(parent, list):
        if not isinstance(key, int):
            raise PatchApplicationError(
                f"Cannot use string key '{key}' to remove from a list"
            )
        if key < 0 or key >= len(parent):
            raise PatchApplicationError(
                f"List index {key} out of range (length {len(parent)})"
            )
        del parent[key]
    elif isinstance(parent, dict):
        if key not in parent:
            raise PatchApplicationError(
                f"Cannot REMOVE non-existent key '{key}' from dict"
            )
        del parent[key]
    else:
        raise PatchApplicationError(
            f"Cannot apply REMOVE to {type(parent).__name__} at key '{key}'"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_patches(
    draft: SemanticIntentDraft,
    patches: list[SemanticPatch],
) -> SemanticIntentDraft:
    """Apply a list of SemanticPatch operations deterministically to a draft.

    Produces a new SemanticIntentDraft with draft_revision incremented by 1.
    The original draft is never mutated (deep copy is made first).

    Each patch that adds or modifies elements carries its own ProvenanceRef
    entries in the patch model itself (Req 15.4). These are preserved in
    the resolution_history of the new draft.

    Args:
        draft: The current SemanticIntentDraft (remains immutable).
        patches: Ordered list of SemanticPatch operations to apply.

    Returns:
        A new SemanticIntentDraft at revision N+1 with patches applied.

    Raises:
        PatchApplicationError: If a patch path is invalid or operation fails.

    Requirements: 4.4, 15.4, 17.1
    """
    # Deep-copy the draft so the original is never mutated (Req 17.1)
    # Use mode="json" to serialize to plain types (strings, ints, etc.) so that
    # patch values (which are plain JSON-like values) can be applied directly
    # without needing to construct enum/model instances.
    draft_dict = copy.deepcopy(draft.model_dump(mode="json"))

    # Increment revision: N → N+1 (Req 17.1)
    draft_dict["draft_revision"] = draft.draft_revision + 1

    # Apply each patch in order (deterministic)
    for patch in patches:
        tokens = _parse_path(patch.path)

        try:
            parent, final_key = _resolve_target(draft_dict, tokens)
        except PatchApplicationError as exc:
            raise PatchApplicationError(
                f"Failed to resolve path '{patch.path}': {exc}",
                patch=patch,
            ) from exc

        try:
            if patch.operation == PatchOp.ADD:
                if patch.value is None:
                    raise PatchApplicationError(
                        f"ADD operation requires a value, got None for path '{patch.path}'",
                        patch=patch,
                    )
                _apply_add(parent, final_key, patch.value)
            elif patch.operation == PatchOp.REPLACE:
                if patch.value is None:
                    raise PatchApplicationError(
                        f"REPLACE operation requires a value, got None for path '{patch.path}'",
                        patch=patch,
                    )
                _apply_replace(parent, final_key, patch.value)
            elif patch.operation == PatchOp.REMOVE:
                _apply_remove(parent, final_key)
            else:
                raise PatchApplicationError(
                    f"Unknown patch operation: {patch.operation}",
                    patch=patch,
                )
        except PatchApplicationError:
            raise
        except Exception as exc:
            raise PatchApplicationError(
                f"Unexpected error applying {patch.operation.value} at '{patch.path}': {exc}",
                patch=patch,
            ) from exc

    # Reconstruct the draft from the modified dict
    # Use model_validate_json to properly deserialize from JSON-typed values
    # (strings for enums, ISO strings for datetimes, etc.)
    try:
        new_draft = SemanticIntentDraft.model_validate_json(json.dumps(draft_dict))
    except Exception as exc:
        raise PatchApplicationError(
            f"Patches produced an invalid draft structure: {exc}"
        ) from exc

    return new_draft
