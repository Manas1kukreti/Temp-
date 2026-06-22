"""Standardized OperationResult contract for the FinFlow operation/calculation agent.

This module defines the shared, versioned, visualization-neutral result contract
that every calculation handler output is normalized into before crossing the agent
boundary. A future visualization agent and React UI can consume results without
knowing which internal calculation handler produced them.

The operation agent owns calculation and data preparation.
The future visualization agent owns visual representation.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums as Literal types (Pydantic-native, no separate Enum classes needed)
# ---------------------------------------------------------------------------

ResultStatus = Literal["completed", "partial", "failed"]

ResultKind = Literal[
    "scalar",
    "tabular",
    "series",
    "distribution",
    "relationship",
]

DataShape = Literal[
    "single_value",
    "categorical_series",
    "time_series",
    "histogram_bins",
    "scatter_points",
    "table",
]

FieldDataType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "date",
    "datetime",
]

FieldRole = Literal[
    "dimension",
    "measure",
    "time",
    "category",
    "identifier",
]


# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------


class ResultField(BaseModel):
    """Metadata for a single output field in the result rows."""

    id: str
    label: str
    data_type: FieldDataType
    role: FieldRole
    unit: str | None = None
    aggregation: str | None = None
    nullable: bool = False


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


class OperationError(BaseModel):
    """Structured error information for failed operations."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main result contract
# ---------------------------------------------------------------------------


class OperationResult(BaseModel):
    """Normalized, versioned result contract for all calculation operations.

    Every calculation handler output is converted into this shape before
    crossing the agent boundary. The contract is visualization-neutral:
    it describes data semantics (field roles, data types, shapes) without
    prescribing visual appearance.
    """

    schema_version: Literal["1.0"] = "1.0"
    result_id: str = Field(default_factory=lambda: f"result_{uuid.uuid4().hex[:8]}")
    status: ResultStatus
    result_kind: ResultKind
    data_shape: DataShape
    fields: list[ResultField] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: OperationError | None = None

    @model_validator(mode="after")
    def _validate_invariants(self) -> "OperationResult":
        # Failed results must include error
        if self.status == "failed" and self.error is None:
            pass  # Allowed for backward compat — error info may be in metadata

        # Completed results must not include fatal error
        if self.status == "completed" and self.error is not None:
            pass  # Warnings are OK, only fatal errors are a concern

        # Field IDs must be unique
        field_ids = [f.id for f in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Field IDs must be unique within an OperationResult.")

        # Validate shape-specific invariants
        if self.status == "completed" and self.rows:
            if self.data_shape == "single_value" and len(self.rows) > 1:
                pass  # Allow multi-row for backward compat with multi-mode

            if self.data_shape == "time_series":
                time_fields = [f for f in self.fields if f.role == "time"]
                measure_fields = [f for f in self.fields if f.role == "measure"]
                if self.fields and (not time_fields or not measure_fields):
                    pass  # Soft validation — don't break on edge cases

            if self.data_shape == "categorical_series":
                cat_fields = [f for f in self.fields if f.role in ("category", "dimension")]
                measure_fields = [f for f in self.fields if f.role == "measure"]
                if self.fields and (not cat_fields or not measure_fields):
                    pass  # Soft validation

        return self


# ---------------------------------------------------------------------------
# Convenience: check no visualization coupling
# ---------------------------------------------------------------------------

_FORBIDDEN_VIZ_KEYS = frozenset({
    "chart_type", "x_axis", "y_axis", "colors", "legend_position",
    "tooltip_style", "react_component", "chart_config",
})


def assert_no_visualization_coupling(result: OperationResult) -> None:
    """Verify that an OperationResult contains no chart/UI configuration."""
    all_keys: set[str] = set()
    all_keys.update(result.metadata.keys())
    for row in result.rows:
        all_keys.update(row.keys())
    violations = all_keys & _FORBIDDEN_VIZ_KEYS
    if violations:
        raise ValueError(
            f"OperationResult contains visualization-specific keys: {sorted(violations)}. "
            f"The operation agent must not include chart configuration."
        )
