import pandas as pd
import time
import uuid
from typing import Dict, Any, Callable

from finflow_agent.state import ExecutionOutput
from finflow_agent.operations.schemas import (
    CleaningOperationPlan, FilterOperationPlan, CalculationOperationPlan,
    VisualizationOperationPlan, ReportingOperationPlan
)
from finflow_agent.operations.errors import UnsupportedOperationError
from finflow_agent.operations.cleaning_handlers import CLEANING_HANDLERS
from finflow_agent.operations.filter_handlers import FILTER_HANDLERS
from finflow_agent.operations.calculation_handlers import CALCULATION_HANDLERS
from finflow_agent.operations.visualization_handlers import VISUALIZATION_HANDLERS
from finflow_agent.operations.reporting_handlers import REPORTING_HANDLERS
from finflow_agent.operations.validators import required_columns_for_operation, validate_columns_exist, hash_operation_params
from finflow_agent.contract_registry import check_action_kind_coverage

def execute_cleaning_plan(df: pd.DataFrame, plan: CleaningOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    for op in plan.operations:
        req_cols = required_columns_for_operation(op)
        validate_columns_exist(output.data, req_cols)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = CLEANING_HANDLERS.get(op.type)
        if not handler:
            raise UnsupportedOperationError(f"No cleaning handler found for {op.type}")
            
        metrics = handler(output.data, op)
        if metrics is None:
            metrics = {}
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        cols_mod = list(set(input_cols) ^ set(output_cols))
        targeted = []
        if hasattr(op, "column") and getattr(op, "column"):
            targeted.append(op.column)
        if hasattr(op, "columns"):
            cols = op.columns
            if cols != "__all_string_columns__":
                if isinstance(cols, str):
                    targeted.append(cols)
                elif isinstance(cols, list):
                    targeted.extend(cols)
        for c in targeted:
            if c in output_cols and c not in cols_mod:
                cols_mod.append(c)
                
        op_warnings = metrics.get("warnings", [])
        if op_warnings:
            output.warnings.extend(op_warnings)
            
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": op.type,
            "type": op.type,
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": op_warnings,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(op),
            "changed_values": initial_rows - len(output.data) if initial_rows != len(output.data) else 0,
            **{k: v for k, v in metrics.items() if k not in ["warnings"]}
        })
        
    output.summary = f"Successfully applied {len(plan.operations)} cleaning operations."
    return output


def _resolve_aggregate_value(df: pd.DataFrame, cond) -> "FilterCondition":
    """Resolve aggregate value references in filter conditions.

    When a condition's value is a dict representing an aggregate function
    (e.g., {"type": "aggregate", "agg_func": "avg", ...} or {"function": "avg", ...}),
    compute the aggregate from the dataframe and return a new condition with the
    resolved scalar value. Returns the original condition unchanged if value is not
    an aggregate reference.
    """
    from finflow_agent.operations.schemas import FilterCondition

    value = cond.value
    if not isinstance(value, dict):
        return cond

    is_aggregate = (
        value.get("type") in {"aggregate", "average", "avg", "sum", "min", "max", "median", "count", "std"}
        or "agg_func" in value
        or "function" in value
    )
    if not is_aggregate:
        return cond

    agg_func = str(
        value.get("agg_func") or value.get("function") or value.get("type") or "avg"
    ).strip().lower()

    field_ref = value.get("field_ref")
    if isinstance(field_ref, dict):
        agg_column = (
            field_ref.get("resolved_column")
            or field_ref.get("reference_text")
            or cond.column
        )
    else:
        agg_column = cond.column

    col_lower_map = {c.lower(): c for c in df.columns}
    resolved_col = col_lower_map.get(agg_column.lower(), agg_column)
    if resolved_col not in df.columns:
        resolved_col = cond.column

    raw_series = df[resolved_col].astype(str).str.replace(r'[$€£¥,]', '', regex=True).str.strip()
    numeric_series = pd.to_numeric(raw_series, errors="coerce")

    agg_map = {
        "avg": numeric_series.mean,
        "mean": numeric_series.mean,
        "average": numeric_series.mean,
        "sum": numeric_series.sum,
        "min": numeric_series.min,
        "max": numeric_series.max,
        "median": numeric_series.median,
        "count": numeric_series.count,
        "std": numeric_series.std,
    }
    compute_fn = agg_map.get(agg_func)
    if compute_fn is None:
        raise UnsupportedOperationError(
            f"Unsupported aggregate function in filter value: {agg_func!r}"
        )

    resolved_value = compute_fn()

    target_col = cond.column
    if target_col not in df.columns:
        target_col = col_lower_map.get(target_col.lower(), target_col)
    if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
        stripped = df[target_col].astype(str).str.replace(r'[$€£¥,]', '', regex=True).str.strip()
        df[target_col] = pd.to_numeric(stripped, errors="coerce")

    return FilterCondition(
        column=target_col,
        operator=cond.operator,
        value=resolved_value,
        value_to=cond.value_to,
        case_sensitive=cond.case_sensitive,
    )


def execute_filter_plan(df: pd.DataFrame, plan: FilterOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    initial_rows = len(output.data)
    input_cols = list(output.data.columns)
    
    started_at = int(time.time() * 1000)
    
    if plan.select_columns:
        validate_columns_exist(output.data, plan.select_columns)
        
    if not plan.conditions:
        if plan.select_columns:
            output.data = output.data[plan.select_columns]
        if plan.limit:
            output.data = output.data.head(plan.limit)
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        cols_mod = list(set(input_cols) ^ set(output_cols))
        
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": "filter_select",
            "type": "filter_select",
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(plan.model_dump())
        })
        output.summary = "No filter conditions applied."
        return output
        
    masks = []
    for cond in plan.conditions:
        validate_columns_exist(output.data, cond.column)

        # Resolve aggregate value references (e.g., {"type": "aggregate", "agg_func": "avg"})
        resolved_cond = _resolve_aggregate_value(output.data, cond)

        handler = FILTER_HANDLERS.get(resolved_cond.operator)
        if not handler:
            raise UnsupportedOperationError(f"No filter handler found for {resolved_cond.operator}")
            
        mask = handler(output.data[resolved_cond.column], resolved_cond)
        masks.append(mask)
        
    if masks:
        final_mask = masks[0]
        if plan.logic == "and":
            for m in masks[1:]:
                final_mask = final_mask & m
        elif plan.logic == "or":
            for m in masks[1:]:
                final_mask = final_mask | m
        output.data = output.data[final_mask]
        
    if plan.select_columns:
        output.data = output.data[plan.select_columns]
        
    if plan.limit:
        output.data = output.data.head(plan.limit)
        
    finished_at = int(time.time() * 1000)
    output_cols = list(output.data.columns)
    cols_mod = list(set(input_cols) ^ set(output_cols))
    if plan.select_columns:
        cols_mod.extend([c for c in plan.select_columns if c not in cols_mod])
        
    output.operations_applied.append({
        "operation_id": f"op_{uuid.uuid4().hex[:8]}",
        "operation_type": "filter",
        "type": "filter",
        "input_row_count": initial_rows,
        "output_row_count": len(output.data),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "columns_modified": cols_mod,
        "warnings": [],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": finished_at - started_at,
        "params_hash": hash_operation_params(plan.model_dump())
    })
    
    output.summary = f"Filtered from {initial_rows} to {len(output.data)} rows using {plan.logic} logic."
    return output

def execute_calculation_plan(df: pd.DataFrame, plan: CalculationOperationPlan) -> ExecutionOutput:
    from finflow_agent.operations.result_builder import normalize_handler_result

    output = ExecutionOutput(data=df.copy())
    operation_results: list[dict] = []
    
    for op in plan.operations:
        req_cols = required_columns_for_operation(op)
        validate_columns_exist(output.data, req_cols)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = CALCULATION_HANDLERS.get(op.type)
        if not handler:
            raise UnsupportedOperationError(f"No calculation handler found for {op.type}")
             
        res = handler(output.data, op)
        if res is None:
            res = {}
            
        if "metrics" in res:
            output.metrics.update(res["metrics"])
        if "df" in res:
            output.data = res["df"]
        if "warnings" in res:
            output.warnings.extend(res["warnings"])

        # Normalize handler result into the shared OperationResult contract
        op_result = normalize_handler_result(res, op)
        operation_results.append(op_result.model_dump(mode="json"))
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        cols_mod = list(set(input_cols) ^ set(output_cols))
        if op.output_column and op.output_column in output_cols and op.output_column not in cols_mod:
            cols_mod.append(op.output_column)
            
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": op.type,
            "type": op.type,
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": res.get("warnings", []),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(op),
            "column": op.column
        })

    # Attach normalized operation results to artifacts
    output.artifacts["operation_results"] = operation_results
    output.summary = f"Successfully calculated {len(plan.operations)} metrics."
    return output

def execute_visualization_plan(df: pd.DataFrame, plan: VisualizationOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    for chart in plan.charts:
        validate_columns_exist(output.data, chart.x)
        validate_columns_exist(output.data, chart.y)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = VISUALIZATION_HANDLERS.get(chart.type)
        if not handler:
            raise UnsupportedOperationError(f"No visualization handler found for {chart.type}")
            
        res = handler(output.data, chart)
        output.artifacts[res["chart_id"]] = res["spec"]
        
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": "visualization",
            "type": "visualization",
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": [],
            "warnings": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(chart),
            "chart_type": chart.type
        })
        
    output.summary = f"Successfully generated {len(plan.charts)} chart specifications."
    return output

def execute_reporting_plan(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, chart_configs: list = None) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    started_at = int(time.time() * 1000)
    initial_rows = len(output.data)
    input_cols = list(output.data.columns)
    
    handler = REPORTING_HANDLERS.get(plan.output_format)
    if not handler:
        raise UnsupportedOperationError(f"No reporting handler found for format {plan.output_format}")
        
    res = handler(output.data, plan, output_dir, file_prefix, chart_configs=chart_configs)
    output.artifacts.update(res)
    
    finished_at = int(time.time() * 1000)
    output_cols = list(output.data.columns)
    
    output.operations_applied.append({
        "operation_id": f"op_{uuid.uuid4().hex[:8]}",
        "operation_type": "reporting",
        "type": "reporting",
        "input_row_count": initial_rows,
        "output_row_count": len(output.data),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "columns_modified": [],
        "warnings": [],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": finished_at - started_at,
        "params_hash": hash_operation_params(plan),
        "format": plan.output_format
    })
    
    output.summary = f"Successfully exported {plan.output_format} report to {res.get('output_file_path')}."
    return output


# ---------------------------------------------------------------------------
# Placeholder handlers for action kinds without dedicated implementations yet
# ---------------------------------------------------------------------------


def execute_project_columns_plan(df: pd.DataFrame, plan: Any) -> ExecutionOutput:
    """Handler for project_columns action kind.

    Selects a subset of columns from the DataFrame.
    """
    raise NotImplementedError(
        "project_columns action handler is not yet implemented. "
        "Add implementation in executor.py."
    )


def execute_drop_columns_plan(df: pd.DataFrame, plan: Any) -> ExecutionOutput:
    """Handler for drop_columns action kind.

    Drops specified columns from the DataFrame.
    """
    raise NotImplementedError(
        "drop_columns action handler is not yet implemented. "
        "Add implementation in executor.py."
    )


def execute_rename_columns_plan(df: pd.DataFrame, plan: Any) -> ExecutionOutput:
    """Handler for rename_columns action kind.

    Renames specified columns in the DataFrame.
    """
    raise NotImplementedError(
        "rename_columns action handler is not yet implemented. "
        "Add implementation in executor.py."
    )


def execute_sort_rows_plan(df: pd.DataFrame, plan: Any) -> ExecutionOutput:
    """Handler for sort_rows action kind.

    Sorts the DataFrame by specified columns.
    """
    raise NotImplementedError(
        "sort_rows action handler is not yet implemented. "
        "Add implementation in executor.py."
    )


def execute_limit_rows_plan(df: pd.DataFrame, plan: Any) -> ExecutionOutput:
    """Handler for limit_rows action kind.

    Limits the number of rows in the DataFrame.
    """
    raise NotImplementedError(
        "limit_rows action handler is not yet implemented. "
        "Add implementation in executor.py."
    )


# ---------------------------------------------------------------------------
# Action Handler Registry — maps action kind strings to handler functions
# ---------------------------------------------------------------------------

ACTION_HANDLERS: Dict[str, Callable] = {
    "clean": execute_cleaning_plan,
    "project_columns": execute_project_columns_plan,
    "drop_columns": execute_drop_columns_plan,
    "rename_columns": execute_rename_columns_plan,
    "filter_rows": execute_filter_plan,
    "sort_rows": execute_sort_rows_plan,
    "limit_rows": execute_limit_rows_plan,
    "calculate": execute_calculation_plan,
    "visualize": execute_visualization_plan,
    "report": execute_reporting_plan,
}

# Import-time coverage check: raises ImportError if any ActionKind member is missing
# or if any unknown action kind is registered.
check_action_kind_coverage(ACTION_HANDLERS)
