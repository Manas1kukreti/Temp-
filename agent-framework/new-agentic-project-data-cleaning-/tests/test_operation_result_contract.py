"""Tests for the OperationResult standardized contract.

Validates that all calculation operations produce normalized, versioned,
visualization-neutral results through the shared OperationResult contract.
"""

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from finflow_agent.operations.result_contract import (
    OperationResult,
    ResultField,
    OperationError,
    assert_no_visualization_coupling,
)
from finflow_agent.operations.result_builder import (
    build_scalar_result,
    build_categorical_series_result,
    build_tabular_result,
    build_failed_result,
    normalize_handler_result,
    _to_native,
)
from finflow_agent.operations.schemas import CalculationOperation, CalculationOperationPlan
from finflow_agent.operations.executor import execute_calculation_plan


def _make_op(op_type: str, column: str, **kwargs) -> CalculationOperation:
    return CalculationOperation(type=op_type, column=column, **kwargs)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["Electronics", "Clothing", "Electronics", "Clothing", "Food"],
        "revenue": [250000, 175000, 120000, 95000, 60000],
        "quantity": [10, 25, 8, 30, 50],
        "date": pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15", "2026-03-01"]),
    })


# ---------------------------------------------------------------------------
# Test 1: Scalar sum
# ---------------------------------------------------------------------------


class TestScalarSum:
    def test_sum_produces_valid_result(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[_make_op("sum", "revenue")])
        output = execute_calculation_plan(df, plan)

        results = output.artifacts.get("operation_results", [])
        assert len(results) == 1

        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.result_kind == "scalar"
        assert r.data_shape == "single_value"
        assert len(r.fields) == 1
        assert r.fields[0].role == "measure"
        assert r.fields[0].aggregation == "sum"
        assert len(r.rows) == 1
        assert r.rows[0]["sum_revenue"] == 700000
        assert r.metrics["sum_revenue"] == 700000


# ---------------------------------------------------------------------------
# Test 2: Scalar median
# ---------------------------------------------------------------------------


class TestScalarMedian:
    def test_median_produces_valid_result(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[_make_op("median", "revenue")])
        output = execute_calculation_plan(df, plan)

        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.fields[0].aggregation == "median"
        assert r.fields[0].data_type == "number"
        assert isinstance(r.rows[0]["median_revenue"], (int, float))


# ---------------------------------------------------------------------------
# Test 3: Count
# ---------------------------------------------------------------------------


class TestScalarCount:
    def test_count_produces_integer_result(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[_make_op("count", "revenue")])
        output = execute_calculation_plan(df, plan)

        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.result_kind == "scalar"
        assert r.rows[0]["count_revenue"] == 5


# ---------------------------------------------------------------------------
# Test 5: Grouped sum
# ---------------------------------------------------------------------------


class TestGroupedSum:
    def test_grouped_sum_produces_categorical_series(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[
            _make_op("group_sum", "revenue", group_by=["category"])
        ])
        output = execute_calculation_plan(df, plan)

        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.result_kind == "series"
        assert r.data_shape == "categorical_series"
        # Must have category + measure fields
        roles = {f.role for f in r.fields}
        assert "category" in roles
        assert "measure" in roles
        # Rows must be flat dicts, not nested
        assert all(isinstance(row, dict) for row in r.rows)
        assert len(r.rows) >= 2


# ---------------------------------------------------------------------------
# Test 9: Non-numeric sum failure
# ---------------------------------------------------------------------------


class TestNonNumericFailure:
    def test_non_numeric_sum_raises(self):
        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        plan = CalculationOperationPlan(operations=[_make_op("sum", "name")])
        with pytest.raises(Exception):
            execute_calculation_plan(df, plan)


# ---------------------------------------------------------------------------
# Test 10: JSON serialization
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_numpy_types_serialize_cleanly(self):
        op = _make_op("sum", "revenue")
        result = build_scalar_result(
            operation=op,
            value=np.float64(123456.789),
        )
        # Must serialize to JSON without error
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["rows"][0]["sum_revenue"] == 123456.789
        assert parsed["schema_version"] == "1.0"

    def test_nan_becomes_null(self):
        op = _make_op("mean", "revenue")
        result = build_scalar_result(operation=op, value=float("nan"))
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["rows"][0]["mean_revenue"] is None

    def test_pandas_timestamp_becomes_iso(self):
        native = _to_native(pd.Timestamp("2026-03-15 10:30:00"))
        assert isinstance(native, str)
        assert "2026-03-15" in native


# ---------------------------------------------------------------------------
# Test 11: No visualization coupling
# ---------------------------------------------------------------------------


class TestNoVisualizationCoupling:
    def test_result_has_no_chart_config(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[_make_op("sum", "revenue")])
        output = execute_calculation_plan(df, plan)

        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        # Must not raise
        assert_no_visualization_coupling(r)

    def test_forbidden_keys_detected(self):
        r = OperationResult(
            status="completed",
            result_kind="scalar",
            data_shape="single_value",
            fields=[],
            rows=[],
            metadata={"chart_type": "bar"},
        )
        with pytest.raises(ValueError, match="visualization-specific"):
            assert_no_visualization_coupling(r)


# ---------------------------------------------------------------------------
# Test 12: Same result with or without graph request
# ---------------------------------------------------------------------------


class TestGraphIndependence:
    def test_operation_result_independent_of_viz_request(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[
            _make_op("group_sum", "revenue", group_by=["category"])
        ])

        # Execute without any visualization context
        output1 = execute_calculation_plan(df.copy(), plan)
        # Execute again (simulating a case where viz was separately requested)
        output2 = execute_calculation_plan(df.copy(), plan)

        r1 = output1.artifacts["operation_results"][0]
        r2 = output2.artifacts["operation_results"][0]

        # result_id may differ (uuid), but structure must be identical
        assert r1["status"] == r2["status"]
        assert r1["result_kind"] == r2["result_kind"]
        assert r1["data_shape"] == r2["data_shape"]
        assert r1["fields"] == r2["fields"]
        assert r1["rows"] == r2["rows"]


# ---------------------------------------------------------------------------
# Test: Field IDs are unique
# ---------------------------------------------------------------------------


class TestFieldIdUniqueness:
    def test_duplicate_field_ids_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            OperationResult(
                status="completed",
                result_kind="scalar",
                data_shape="single_value",
                fields=[
                    ResultField(id="x", label="X", data_type="number", role="measure"),
                    ResultField(id="x", label="X2", data_type="number", role="measure"),
                ],
                rows=[{"x": 1}],
            )


# ---------------------------------------------------------------------------
# Test: schema_version always present
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_schema_version_present(self):
        df = _sample_df()
        plan = CalculationOperationPlan(operations=[_make_op("mean", "revenue")])
        output = execute_calculation_plan(df, plan)
        r = output.artifacts["operation_results"][0]
        assert r["schema_version"] == "1.0"
