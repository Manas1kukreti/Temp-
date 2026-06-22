"""Semantic patch models for FinFlow's bounded repair stage.

Defines typed patch operations applied against declared draft paths. Patches are
the output of the Semantic Repair stage and represent the only permissible
modifications to a SemanticIntentDraft during automated repair.

Requirements: 4.3, 4.4
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.provenance import ProvenanceRef


class PatchOp(str, Enum):
    """Permitted patch operation types for semantic repair."""

    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"


class SemanticPatch(BaseModel):
    """Typed operation applied against a declared draft path.

    Each patch records which validator failure triggered it and includes
    provenance for auditability. Only typed operations (add, replace, remove)
    are permitted.

    Requirements: 4.3 - typed patches only (add, replace, remove)
    """

    model_config = ConfigDict(strict=True)

    operation: PatchOp
    path: str  # JSON-path-like reference into draft
    value: Any = None  # For add/replace; None for remove
    reason: str
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    source_failure: str  # Which validator failure triggered this patch
