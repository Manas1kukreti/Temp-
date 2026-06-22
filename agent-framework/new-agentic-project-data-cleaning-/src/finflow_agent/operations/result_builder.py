"""Result builders: normalize calculation handler outputs into OperationResult.

Each builder function converts a specific handler return shape into the
shared OperationResult contract. The executor calls the appropriate builder
after each handler completes.

This is the single normalization boundary — handlers may continue returning
their internal structures, but everything that crosses the agent boundary
passes through these builders.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from finflow_agent.operations.result_contract import (
    DataShape,
    FieldRole,
    OperationError,
    OperationResult,
    ResultField,
    ResultKind,
)
from finflow_agent.operations.schemas import CalculationOperation


# ---------------------------------------------------------------------------
# JSON-safe serialization helpers
# ---------------------------------------------------------------------------


def _to_native(value: Any) -> Any:
    """Convert NumPy/Pandas scalars to native Python types."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure all values in a row dict are JSON-serializable."""
    return {k: _to_native(v) for k, v in row.items()}


def _infer_field_data_type(value: Any) -> str:
    """Infer the FieldDataType from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _infer_unit(column_name: str) -> str | None:
    """Infer unit from column name heuristics."""
    currency_keywords = [
        "revenue", "price", "amount", "cost", "sales", "profit",
        "total", "balance", "value", "income", "loan",
    ]
    col_lower = column_name.lower()
    if any(kw in col_lower for kw in currency_keywords):
        return "INR"
    if "percent" in col_lower or "pct" in col_lower or "rate" in col_lower:
        return "percent"
    return None


def _make_result_id(operation_type: str, column: str) -> str:
    """Generate a stable result_id from the operation and column."""
    safe_col = column.replace(" ", "_").lower()
    return f"{operation_type}_{safe_col}"


# ---------------------------------------------------------------------------
# Scalar result builder
# ---------------------------------------------------------------------------


def build_scalar_result(
    *,
    operation: CalculationOperation,
    value: Any,
    warnings: list[str] | None = None,
) -> OperationResult:
    """Build a scalar OperationResult (sum, mean, median, min, max, count, etc.)."""
    out_col = operation.output_column or f"{operation.type}_{operation.column}"
    native_value = _to_native(value)
    data_type = _infer_field_data_type(native_value) if native_value is not None else "number"
    unit = _infer_unit(out_col)

    return OperationResult(
        result_id=_make_result_id(operation.type, operation.column),
        status="completed",
        result_kind="scalar",
        data_shape="single_value",
        fields=[
            ResultField(
                id=out_col,
                label=out_col.replace("_", " ").title(),
                data_type=data_type,
                role="measure",
                unit=unit,
                aggregation=operation.type,
                nullable=False,
            )
        ],
        rows=[{out_col: native_value}],
        metrics={out_col: native_value},
        metadata={
            "operation": operation.type,
            "source_fields": [operation.column],
            "null_policy": "excluded",
        },
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Categorical series (grouped aggregation) result builder
# ---------------------------------------------------------------------------


def build_categorical_series_result(
    *,
    operation: CalculationOperation,
    df: pd.DataFrame,
    warnings: list[str] | None = None,
) -> OperationResult:
    """Build a categorical series OperationResult from grouped aggregation."""
    out_col = operation.output_column or f"{operation.type.replace('group_', '')}_{operation.column}"
    group_cols = operation.group_by or []

    fields: list[ResultField] = []
    for gcol in group_cols:
        fields.append(ResultField(
            id=gcol,
            label=gcol.replace("_", " ").title(),
            data_type="string",
            role="category",
            nullable=False,
        ))

    # Find the measure column in the dataframe
    measure_col = out_col if out_col in df.columns else operation.column
    if measure_col not in df.columns:
        # Try to find any numeric column that isn't in group_cols
        for c in df.columns:
            if c not in group_cols and pd.api.types.is_numeric_dtype(df[c]):
                measure_col = c
                break

    agg_type = operation.type.replace("group_", "")
    unit = _infer_unit(measure_col)
    fields.append(ResultField(
        id=measure_col,
        label=measure_col.replace("_", " ").title(),
        data_type="number",
        role="measure",
        unit=unit,
        aggregation=agg_type,
        nullable=False,
    ))

    # Build rows from only the declared field IDs
    declared_ids = [f.id for f in fields]
    rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for fid in declared_ids:
            if fid in row.index:
                row_dict[fid] = _to_native(row[fid])
        rows.append(row_dict)

    result_id = f"{agg_type}_{operation.column}_by_{'_'.join(group_cols)}"

    return OperationResult(
        result_id=result_id,
        status="completed",
        result_kind="series",
        data_shape="categorical_series",
        fields=fields,
        rows=rows,
        metadata={
            "operation": agg_type,
            "source_fields": [operation.column],
            "group_by": group_cols,
            "null_policy": "excluded",
        },
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Tabular/DataFrame result builder (running_total, pct_change, diff, ratio, abs)
# ---------------------------------------------------------------------------


def build_tabular_result(
    *,
    operation: CalculationOperation,
    df: pd.DataFrame,
    warnings: list[str] | None = None,
) -> OperationResult:
    """Build a tabular OperationResult from DataFrame-mutating operations."""
    fields: list[ResultField] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data_type = "number"
            role: FieldRole = "measure"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            data_type = "datetime"
            role = "time"
        else:
            data_type = "string"
            role = "dimension"

        fields.append(ResultField(
            id=col,
            label=col.replace("_", " ").title(),
            data_type=data_type,
            role=role,
            unit=_infer_unit(col),
            nullable=bool(df[col].isna().any()),
        ))

    declared_ids = [f.id for f in fields]
    rows = []
    for _, row in df.head(1000).iterrows():  # Cap at 1000 for result payload
        rows.append({fid: _to_native(row[fid]) for fid in declared_ids if fid in row.index})

    return OperationResult(
        result_id=_make_result_id(operation.type, operation.column),
        status="completed",
        result_kind="tabular",
        data_shape="table",
        fields=fields,
        rows=rows,
        metadata={
            "operation": operation.type,
            "source_fields": [operation.column],
            "total_rows": len(df),
            "null_policy": "excluded",
        },
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Failed result builder
# ---------------------------------------------------------------------------


def build_failed_result(
    *,
    operation: CalculationOperation,
    error_code: str,
    error_message: str,
    details: dict[str, Any] | None = None,
) -> OperationResult:
    """Build a failed OperationResult with structured error."""
    return OperationResult(
        result_id=_make_result_id(operation.type, operation.column),
        status="failed",
        result_kind="scalar",
        data_shape="single_value",
        fields=[],
        rows=[],
        metadata={
            "operation": operation.type,
            "source_fields": [operation.column],
        },
        error=OperationError(
            code=error_code,
            message=error_message,
            details=details or {},
        ),
    )


# ---------------------------------------------------------------------------
# Universal normalizer: handler result → OperationResult
# ---------------------------------------------------------------------------


def normalize_handler_result(
    handler_result: dict[str, Any],
    operation: CalculationOperation,
) -> OperationResult:
    """Convert a raw calculation handler result into the OperationResult contract.

    This is the single normalization point. All handlers return their internal
    structures, and this function converts them into the shared contract before
    the result crosses the agent boundary.
    """
    warnings = handler_result.get("warnings", [])

    # Scalar result (metrics dict)
    if "metrics" in handler_result:
        metrics = handler_result["metrics"]
        out_col = operation.output_column or f"{operation.type}_{operation.column}"
        value = metrics.get(out_col, next(iter(metrics.values()), None))
        return build_scalar_result(
            operation=operation,
            value=value,
            warnings=warnings,
        )

    # DataFrame result
    if "df" in handler_result:
        df = handler_result["df"]
        # Grouped aggregation (group_sum, group_mean, group_count)
        if operation.type in ("group_sum", "group_mean", "group_count") and operation.group_by:
            return build_categorical_series_result(
                operation=operation,
                df=df,
                warnings=warnings,
            )
        # Quarterly aggregation (quarterly_sum, quarterly_mean, quarterly_count)
        if operation.type in ("quarterly_sum", "quarterly_mean", "quarterly_count"):
            return build_categorical_series_result(
                operation=operation,
                df=df,
                warnings=warnings,
            )
        # Tabular operations (running_total, pct_change, difference, ratio, abs)
        return build_tabular_result(
            operation=operation,
            df=df,
            warnings=warnings,
        )

    # Fallback: treat as empty completed result
    return OperationResult(
        result_id=_make_result_id(operation.type, operation.column),
        status="completed",
        result_kind="scalar",
        data_shape="single_value",
        fields=[],
        rows=[],
        metadata={"operation": operation.type, "source_fields": [operation.column]},
        warnings=warnings,
    )
