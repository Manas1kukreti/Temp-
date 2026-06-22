"""Tests for advanced calculation operations: conditional percentage and quarterly aggregation."""

import pandas as pd
import pytest

from finflow_agent.operations.schemas import CalculationOperation, CalculationOperationPlan
from finflow_agent.operations.executor import execute_calculation_plan
from finflow_agent.operations.result_contract import OperationResult


def _employees_df() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": range(1, 11),
        "gender": ["female", "male", "female", "female", "male", "female", "male", "female", "male", "male"],
        "marital_status": ["single", "married", "married", "divorced", "single", "single", "married", "divorced", "single", "married"],
        "department": ["eng", "eng", "sales", "eng", "sales", "hr", "hr", "eng", "sales", "eng"],
        "salary": [70000, 85000, 65000, 72000, 90000, 68000, 78000, 71000, 88000, 92000],
    })


def _revenue_df() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": range(1, 13),
        "date": pd.to_datetime([
            "2026-01-15", "2026-02-20", "2026-03-10",
            "2026-04-05", "2026-05-18", "2026-06-22",
            "2026-07-01", "2026-08-14", "2026-09-30",
            "2026-10-12", "2026-11-25", "2026-12-31",
        ]),
        "revenue": [10000, 12000, 15000, 18000, 20000, 22000, 25000, 28000, 30000, 35000, 40000, 45000],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })


# ---------------------------------------------------------------------------
# Conditional Percentage Tests
# ---------------------------------------------------------------------------


class TestConditionalPercentage:
    def test_percentage_of_female_single(self):
        """Percentage of female employees who are single."""
        df = _employees_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="conditional_percentage",
                column="marital_status",
                filter_column="marital_status",
                filter_value="single",
                denominator_filter_column="gender",
                denominator_filter_value="female",
            )
        ])
        output = execute_calculation_plan(df, plan)
        # 5 females, 2 are single → 40%
        assert output.metrics["pct_marital_status_single"] == 40.0

    def test_percentage_of_total_married(self):
        """Percentage of total employees who are married."""
        df = _employees_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="conditional_percentage",
                column="marital_status",
                filter_column="marital_status",
                filter_value="married",
            )
        ])
        output = execute_calculation_plan(df, plan)
        # 10 total, 4 married → 40%
        assert output.metrics["pct_marital_status_married"] == 40.0

    def test_percentage_of_female_divorced(self):
        """Percentage of female employees who are divorced."""
        df = _employees_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="conditional_percentage",
                column="marital_status",
                filter_column="marital_status",
                filter_value="divorced",
                denominator_filter_column="gender",
                denominator_filter_value="female",
            )
        ])
        output = execute_calculation_plan(df, plan)
        # 5 females, 2 divorced → 40%
        assert output.metrics["pct_marital_status_divorced"] == 40.0

    def test_result_contract_compliance(self):
        """Verify the result conforms to OperationResult contract."""
        df = _employees_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="conditional_percentage",
                column="marital_status",
                filter_column="marital_status",
                filter_value="single",
            )
        ])
        output = execute_calculation_plan(df, plan)
        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.result_kind == "scalar"


# ---------------------------------------------------------------------------
# Quarterly Aggregation Tests
# ---------------------------------------------------------------------------


class TestQuarterlyAggregation:
    def test_quarterly_sum(self):
        """Quarterly revenue totals."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_sum",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        result_df = output.data
        # Should have 4 quarters
        assert len(result_df) == 4
        assert "quarter" in result_df.columns
        # Q1 = 10000 + 12000 + 15000 = 37000
        q1 = result_df[result_df["quarter"].str.contains("Q1")].iloc[0]
        assert q1["quarterly_sum_revenue"] == 37000

    def test_quarterly_mean(self):
        """Quarterly average revenue."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_mean",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        result_df = output.data
        assert len(result_df) == 4
        # Q1 mean = (10000 + 12000 + 15000) / 3 ≈ 12333.33
        q1 = result_df[result_df["quarter"].str.contains("Q1")].iloc[0]
        assert abs(q1["quarterly_mean_revenue"] - 12333.33) < 1

    def test_quarterly_count(self):
        """Quarterly transaction count."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_count",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        result_df = output.data
        assert len(result_df) == 4
        # Each quarter has 3 transactions
        assert all(result_df["quarterly_count_revenue"] == 3)

    def test_chronological_ordering(self):
        """Results must be in chronological order."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_sum",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        result_df = output.data
        quarters = result_df["quarter"].tolist()
        assert quarters == sorted(quarters)

    def test_period_start_present(self):
        """Results must have machine-sortable period_start."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_sum",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        result_df = output.data
        assert "period_start" in result_df.columns
        assert result_df["period_start"].iloc[0] == "2026-01-01"

    def test_result_contract_compliance(self):
        """Verify quarterly results conform to OperationResult contract."""
        df = _revenue_df()
        plan = CalculationOperationPlan(operations=[
            CalculationOperation(
                type="quarterly_sum",
                column="revenue",
                date_column="date",
            )
        ])
        output = execute_calculation_plan(df, plan)
        results = output.artifacts["operation_results"]
        r = OperationResult.model_validate(results[0])
        assert r.status == "completed"
        assert r.result_kind == "series"
        assert r.data_shape == "categorical_series"
        assert len(r.rows) == 4
