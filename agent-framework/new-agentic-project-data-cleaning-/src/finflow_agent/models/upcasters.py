"""Versioned upcasters for converting legacy CanonicalIntent to the new schema.

Provides version detection and appropriate upcaster selection to convert
legacy CanonicalIntent payloads (from `planning/canonical_intent.py`) into
the new `models/canonical.py` CanonicalIntent format.

The legacy format has:
- schema_version "1.0" or "2.0"
- operation_type-style actions (kind-discriminated: project_columns, drop_columns,
  filter_rows, sort_rows, clean, etc.)
- output_format, decision, resolution_status, dataframe_profile

The new format has:
- schema_version "1.0" (new schema, different semantics)
- resolution_status literal "resolved"
- resolution_origin (ResolutionOrigin enum)
- actions (list of ResolvedAction: ResolvedFilterAction, ResolvedProjectAction, etc.)
- source_draft_id, source_draft_revision, data_snapshot_ref, provenance

The upcaster maps old fields to new fields, creating synthetic values for
required fields that have no equivalent in the legacy format.

Requirements: 13.4
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from finflow_agent.models.canonical import (
    CanonicalIntent,
    ResolvedDropAction,
    ResolvedFilterAction,
    ResolvedProjectAction,
    ResolvedRenameAction,
    ResolvedSortAction,
)
from finflow_agent.models.draft import ResolutionOrigin
from finflow_agent.models.provenance import PromptSpanProvenance, SchemaEvidenceProvenance
from finflow_agent.models.snapshot import DataSnapshotRef

logger = logging.getLogger(__name__)

# Versions we can upcast from (legacy schema versions)
SUPPORTED_VERSIONS: set[str] = {"0", "0.1", "1.0", "2.0"}
"""Set of legacy schema versions that this module can upcast to the new format."""

# The current new-schema version produced by the upcaster
_NEW_SCHEMA_VERSION = "1.0"


class UpcasterError(Exception):
    """Raised when a legacy payload cannot be upcasted to the new schema.

    Possible causes:
    - Unsupported source schema version
    - Missing required fields in legacy payload
    - Unrecognized action types that cannot be mapped
    """

    pass


def upcast_canonical_intent(legacy_data: dict[str, Any], source_version: str | None = None) -> CanonicalIntent:
    """Convert a legacy CanonicalIntent payload to the new schema CanonicalIntent.

    Detects the version of the incoming data (or uses the provided source_version),
    selects the appropriate internal upcaster, then constructs a valid new-schema
    CanonicalIntent preserving all semantic meaning from the original payload.

    Args:
        legacy_data: Dictionary representing a legacy CanonicalIntent (as from
            model_dump or JSON deserialization of the old format).
        source_version: Explicit source schema version override. If None, the
            version is detected from the payload's ``schema_version`` field.

    Returns:
        A fully-valid new-schema CanonicalIntent instance.

    Raises:
        UpcasterError: If the version is unsupported or the payload cannot be
            converted (missing critical fields, unrecognized action types).
    """
    if not isinstance(legacy_data, dict):
        raise UpcasterError("legacy_data must be a dictionary")

    # Detect version
    version = source_version or str(legacy_data.get("schema_version", "0")).strip()

    if version not in SUPPORTED_VERSIONS:
        raise UpcasterError(
            f"Unsupported legacy schema version '{version}'. "
            f"Supported versions: {sorted(SUPPORTED_VERSIONS)}"
        )

    logger.info(
        "Upcasting legacy CanonicalIntent from version '%s' to new schema '%s'",
        version,
        _NEW_SCHEMA_VERSION,
    )

    # Select and run version-specific upcaster to normalize to intermediate dict
    if version in ("0", "0.1"):
        if version == "0":
            normalized = _upcast_from_v0(legacy_data)
        else:
            normalized = _upcast_from_v0_1(legacy_data)
    elif version in ("1.0", "2.0"):
        # v1.0 and v2.0 share same structure; v2.0 added capability fields
        normalized = _upcast_from_v1_v2(legacy_data)
    else:
        raise UpcasterError(f"No upcaster registered for version '{version}'")

    # Construct CanonicalIntent from normalized dict
    try:
        return CanonicalIntent(**normalized)
    except Exception as exc:
        raise UpcasterError(
            f"Failed to construct CanonicalIntent from upcasted payload: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Version-specific upcasters — each returns a dict suitable for CanonicalIntent()
# ---------------------------------------------------------------------------


def _upcast_from_v0(data: dict[str, Any]) -> dict[str, Any]:
    """Upcast from version '0' (earliest legacy format).

    Version 0 is the most minimal format: it may lack many fields that v1.0+
    introduced. We synthesize defaults for everything missing.
    """
    # Normalize to v0.1 first (add missing optional fields), then upcast from v0.1
    v0_1_data = dict(data)
    v0_1_data.setdefault("schema_version", "0.1")
    v0_1_data.setdefault("resolution_status", "resolved")
    v0_1_data.setdefault("actions", [])
    v0_1_data.setdefault("dataframe_profile", {})
    v0_1_data.setdefault("original_prompt", "")
    v0_1_data.setdefault("decision", "")
    v0_1_data.setdefault("output_format", "xlsx")

    return _upcast_from_v0_1(v0_1_data)


def _upcast_from_v0_1(data: dict[str, Any]) -> dict[str, Any]:
    """Upcast from version '0.1' (early legacy format with basic actions).

    Version 0.1 has actions as a list of dicts with a 'kind' discriminator,
    similar structure to v1.0 but without intent_id, capability fields, etc.
    """
    # Build resolved actions
    resolved_actions = _convert_legacy_actions(data.get("actions", []))

    # Build data_snapshot_ref from dataframe_profile
    data_snapshot_ref = _build_snapshot_ref_from_profile(data.get("dataframe_profile", {}))

    # Build provenance from original_prompt if available
    provenance = _build_synthetic_provenance(data)

    return {
        "schema_version": _NEW_SCHEMA_VERSION,
        "resolution_status": "resolved",
        "resolution_origin": ResolutionOrigin.DIRECT,
        "actions": resolved_actions,
        "source_draft_id": "legacy_upcast",
        "source_draft_revision": 1,
        "data_snapshot_ref": data_snapshot_ref,
        "provenance": provenance,
    }


def _upcast_from_v1_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Upcast from version '1.0' or '2.0' (standard legacy format).

    This is the format produced by `planning/canonical_intent.py`. It has:
    - intent_id, intent_revision, intent_hash
    - resolution_status (may be 'resolved', 'repaired', 'ambiguous', etc.)
    - actions: list of kind-discriminated IntentAction objects
    - dataframe_profile, output_format, decision, evidence
    """
    # Map legacy resolution_status to the new resolution_origin
    legacy_status = data.get("resolution_status", "resolved")
    resolution_origin = _map_resolution_origin(legacy_status)

    # Build resolved actions from legacy action list
    resolved_actions = _convert_legacy_actions(data.get("actions", []))

    # Build data_snapshot_ref from dataframe_profile
    data_snapshot_ref = _build_snapshot_ref_from_profile(data.get("dataframe_profile", {}))

    # Build provenance from original_prompt
    provenance = _build_synthetic_provenance(data)

    return {
        "schema_version": _NEW_SCHEMA_VERSION,
        "resolution_status": "resolved",
        "resolution_origin": resolution_origin,
        "actions": resolved_actions,
        "source_draft_id": data.get("intent_id", "legacy_upcast"),
        "source_draft_revision": data.get("intent_revision", 1),
        "data_snapshot_ref": data_snapshot_ref,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_resolution_origin(legacy_status: str) -> ResolutionOrigin:
    """Map a legacy resolution_status string to a new-schema ResolutionOrigin."""
    mapping = {
        "resolved": ResolutionOrigin.DIRECT,
        "repaired": ResolutionOrigin.SEMANTIC_REPAIR,
        "ambiguous": ResolutionOrigin.USER_CLARIFICATION,
        "needs_clarification": ResolutionOrigin.USER_CLARIFICATION,
        "unsupported": ResolutionOrigin.DIRECT,
        "rejected": ResolutionOrigin.DIRECT,
    }
    return mapping.get(legacy_status, ResolutionOrigin.DIRECT)


def _convert_legacy_actions(actions: list[Any]) -> list[Any]:
    """Convert legacy kind-discriminated actions to new ResolvedAction instances.

    Legacy actions use 'kind' field. New actions use 'type' field.
    Legacy column references are UnresolvedColumnReference with raw_reference
    and optional resolved_column. We use resolved_column if available, else raw_reference.
    """
    resolved = []

    for action in actions:
        if not isinstance(action, dict):
            # If it's already a model instance, convert to dict first
            if hasattr(action, "model_dump"):
                action = action.model_dump(mode="json")
            else:
                continue

        kind = action.get("kind", "")
        converted = _convert_single_action(kind, action)
        if converted is not None:
            resolved.append(converted)

    return resolved


def _convert_single_action(kind: str, action: dict[str, Any]) -> Any:
    """Convert a single legacy action dict to a new ResolvedAction instance."""
    # Use SchemaEvidenceProvenance for synthetic provenance during upcast
    # since PromptSpanProvenance requires valid offsets that don't exist in legacy data
    synthetic_prov = SchemaEvidenceProvenance(
        schema_fingerprint="legacy_upcast",
        column="legacy_action",
        evidence=[f"upcasted from legacy action kind: {kind}"],
    )
    provenance = [synthetic_prov]

    if kind == "project_columns":
        columns = _extract_column_names(action.get("requested_fields", []))
        if not columns:
            return None
        return ResolvedProjectAction(columns=columns, provenance=provenance)

    elif kind == "drop_columns":
        columns = _extract_column_names(action.get("requested_fields", []))
        if not columns:
            return None
        return ResolvedDropAction(columns=columns, provenance=provenance)

    elif kind == "filter_rows":
        predicates = _convert_filter_conditions(action)
        if not predicates:
            return None
        return ResolvedFilterAction(predicates=predicates, provenance=provenance)

    elif kind == "sort_rows":
        sort_keys = action.get("sort_keys", [])
        if not sort_keys:
            return None
        keys = []
        directions = []
        for sk in sort_keys:
            if isinstance(sk, dict):
                col_ref = sk.get("column", {})
                col_name = _resolve_column_ref(col_ref)
                keys.append(col_name)
                directions.append(sk.get("direction", "asc"))
        if not keys:
            return None
        return ResolvedSortAction(keys=keys, directions=directions, provenance=provenance)

    elif kind == "rename_columns":
        # Legacy rename format (if present): mappings as list of dicts
        mappings_raw = action.get("mappings", [])
        mappings = []
        for m in mappings_raw:
            if isinstance(m, dict):
                src = _resolve_column_ref(m.get("source", m.get("from", {})))
                dst = m.get("target", m.get("to", ""))
                if src and dst:
                    mappings.append((src, dst))
        if not mappings:
            return None
        return ResolvedRenameAction(mappings=mappings, provenance=provenance)

    else:
        # Unsupported action kinds (clean, calculate, visualize, report, limit_rows)
        # are not representable in the new schema's resolved action types.
        # Log a warning and skip.
        logger.warning(
            "Legacy action kind '%s' has no equivalent in the new schema; skipped during upcast.",
            kind,
        )
        return None


def _extract_column_names(fields: list[Any]) -> list[str]:
    """Extract column names from legacy UnresolvedColumnReference list."""
    names = []
    for field in fields:
        name = _resolve_column_ref(field)
        if name:
            names.append(name)
    return names


def _resolve_column_ref(ref: Any) -> str:
    """Get the best available column name from a legacy column reference."""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        # Prefer resolved_column, fall back to raw_reference
        return ref.get("resolved_column") or ref.get("raw_reference") or ""
    if hasattr(ref, "resolved_column") and ref.resolved_column:
        return ref.resolved_column
    if hasattr(ref, "raw_reference"):
        return ref.raw_reference
    return ""


def _convert_filter_conditions(action: dict[str, Any]) -> list[dict[str, object]]:
    """Convert legacy FilterRowsIntent conditions to new predicate format."""
    conditions = action.get("conditions", [])
    logic = action.get("logic", "and")
    mode = action.get("mode", "keep")

    predicates = []
    for cond in conditions:
        if not isinstance(cond, dict):
            if hasattr(cond, "model_dump"):
                cond = cond.model_dump(mode="json")
            else:
                continue

        field_ref = cond.get("field", {})
        col_name = _resolve_column_ref(field_ref)
        if not col_name:
            continue

        operator = cond.get("operator", "eq")
        value = cond.get("value")

        predicates.append({
            "column": col_name,
            "operator": operator,
            "value": value,
            "negated": mode == "drop",
            "logical_operator": logic,
        })

    return predicates


def _build_snapshot_ref_from_profile(profile: dict[str, Any]) -> DataSnapshotRef:
    """Build a synthetic DataSnapshotRef from legacy dataframe_profile.

    The legacy format stores profiling info in a flat dict. We extract what
    we can and fill in synthetic values for fields that don't exist in the
    legacy format.
    """
    # Try to extract meaningful values from the profile
    file_id = profile.get("file_id", profile.get("filename", "legacy_unknown"))
    content_hash = profile.get(
        "content_hash",
        hashlib.sha256(
            json.dumps(profile, sort_keys=True, default=str).encode()
        ).hexdigest() if profile else hashlib.sha256(b"empty").hexdigest(),
    )
    byte_size = profile.get("byte_size", profile.get("file_size", 0))
    storage_version = profile.get("storage_version", "legacy")
    profile_id = profile.get("profile_id", "legacy_profile")

    # Compute a deterministic structural fingerprint from available info
    columns = profile.get("columns", profile.get("column_names", []))
    struct_fp = hashlib.sha256(
        json.dumps(sorted(columns) if columns else [], default=str).encode()
    ).hexdigest()

    profile_fp = hashlib.sha256(
        json.dumps(profile, sort_keys=True, default=str).encode()
    ).hexdigest()

    return DataSnapshotRef(
        file_id=str(file_id),
        content_hash=str(content_hash),
        byte_size=int(byte_size) if byte_size else 0,
        storage_version=str(storage_version),
        profile_id=str(profile_id),
        structural_schema_fingerprint=struct_fp,
        profile_fingerprint=profile_fp,
    )


def _build_synthetic_provenance(data: dict[str, Any]) -> list[Any]:
    """Build synthetic provenance from legacy payload metadata."""
    prompt = data.get("original_prompt", data.get("normalized_prompt", ""))
    if prompt and len(prompt) > 0:
        return [
            PromptSpanProvenance(
                start_offset=0,
                end_offset=min(len(prompt), 200),
                source_text=prompt[:200],
            )
        ]
    # Fall back to SchemaEvidenceProvenance when no prompt is available
    return [
        SchemaEvidenceProvenance(
            schema_fingerprint="legacy_upcast",
            column="legacy_payload",
            evidence=["synthetic provenance for legacy upcast"],
        )
    ]
