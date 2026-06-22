from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.contract_registry import CanonicalOperator


CANONICAL_INTENT_SCHEMA_VERSION = "2.0"
CANONICAL_INTENT_ENVELOPE_VERSION = "1.0"
CANONICAL_INTENT_CAPABILITY_VERSION = "agent.capability.1"
SUPPORTED_CANONICAL_OUTPUT_FORMATS = {"xlsx", "csv", "json", "txt"}
SUPPORTED_CANONICAL_ACTIONS = {
    "clean",
    "project_columns",
    "drop_columns",
    "rename_columns",
    "filter_rows",
    "sort_rows",
    "limit_rows",
    "calculate",
    "visualize",
    "report",
}


class UnresolvedColumnReference(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_reference: str
    resolved_column: str | None = None
    resolution_method: str | None = None
    selection_mode: Literal["single", "semantic_family", "ambiguous"] | None = None
    resolved_columns: list[str] = Field(default_factory=list)
    candidate_columns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class FilterCondition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: UnresolvedColumnReference
    operator: CanonicalOperator
    value: Any


class ProjectColumnsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["project_columns"]
    requested_fields: list[UnresolvedColumnReference]


class DropColumnsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["drop_columns"]
    requested_fields: list[UnresolvedColumnReference]


class FilterRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["filter_rows"]
    mode: Literal["keep", "drop"] = "keep"
    conditions: list[FilterCondition] = Field(default_factory=list)
    logic: Literal["and", "or"] = "and"


class CleaningIntentOperation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class CleanIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["clean"]
    mode: Literal["safe_default", "explicit"] = "safe_default"
    operations: list[CleaningIntentOperation] = Field(default_factory=list)


class SortKey(BaseModel):
    model_config = ConfigDict(extra="ignore")

    column: UnresolvedColumnReference
    direction: Literal["asc", "desc"] = "asc"


class SortRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["sort_rows"]
    sort_keys: list[SortKey] = Field(default_factory=list)


class LimitRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["limit_rows"]
    limit: int


class CalculateIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["calculate"]
    operations: list[str] = Field(default_factory=list)


class VisualizeIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["visualize"]
    chart_type: str | None = None
    fields: list[UnresolvedColumnReference] = Field(default_factory=list)


class ReportIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["report"]
    sections: list[str] = Field(default_factory=list)


IntentAction = Annotated[
    ProjectColumnsIntent
    | DropColumnsIntent
    | FilterRowsIntent
    | CleanIntent
    | SortRowsIntent
    | LimitRowsIntent
    | CalculateIntent
    | VisualizeIntent
    | ReportIntent,
    Field(discriminator="kind"),
]


class CanonicalIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = CANONICAL_INTENT_SCHEMA_VERSION
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent_revision: int = 1
    intent_hash: str = ""
    parent_intent_id: str | None = None
    original_prompt: str = ""
    normalized_prompt: str = ""
    resolution_status: Literal[
        "resolved",
        "repaired",
        "ambiguous",
        "needs_clarification",
        "unsupported",
        "rejected",
    ] = "resolved"
    decision: str = ""
    evidence: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    actions: list[IntentAction] = Field(default_factory=list)
    output_format: Literal["xlsx", "csv", "json", "txt"] = "xlsx"
    assumptions: list[str] = Field(default_factory=list)
    repair_notes: list[str] = Field(default_factory=list)
    dataframe_profile: dict[str, Any] = Field(default_factory=dict)
    capability_version: str = CANONICAL_INTENT_CAPABILITY_VERSION
    capability_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    grounded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CanonicalIntentEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = CANONICAL_INTENT_ENVELOPE_VERSION
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent_revision: int = 1
    intent_hash: str = ""
    parent_intent_id: str | None = None
    intent: CanonicalIntent
    original_instruction: str = ""
    extractor_version: str = "1.0"
    normalizer_version: str = "1.0"
    grounding_version: str = "1.0"
    capability_version: str = CANONICAL_INTENT_CAPABILITY_VERSION
    capability_snapshot: dict[str, Any] = Field(default_factory=dict)
    repair_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    grounded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_intent_hash(intent: CanonicalIntent | dict[str, Any]) -> str:
    payload = intent.model_dump(mode="json") if isinstance(intent, CanonicalIntent) else dict(intent)
    for key in ("intent_id", "intent_revision", "intent_hash", "parent_intent_id", "created_at", "grounded_at"):
        payload.pop(key, None)
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def upcast_canonical_intent_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("canonical intent payload must be a dictionary")
    version = str(payload.get("schema_version") or CANONICAL_INTENT_SCHEMA_VERSION).strip()
    if version == CANONICAL_INTENT_SCHEMA_VERSION:
        return payload
    if version != "1.0":
        raise ValueError(f"Unsupported canonical intent schema version: {version}")
    upgraded = dict(payload)
    upgraded["schema_version"] = CANONICAL_INTENT_SCHEMA_VERSION
    upgraded.setdefault("intent_id", str(uuid.uuid4()))
    upgraded.setdefault("intent_revision", 1)
    upgraded.setdefault("intent_hash", "")
    upgraded.setdefault("parent_intent_id", None)
    upgraded.setdefault("capability_version", CANONICAL_INTENT_CAPABILITY_VERSION)
    upgraded.setdefault("capability_snapshot", {})
    upgraded.setdefault("created_at", datetime.now(UTC))
    upgraded.setdefault("grounded_at", datetime.now(UTC))
    return upgraded
