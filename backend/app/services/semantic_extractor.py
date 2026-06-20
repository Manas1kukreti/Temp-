"""Constrained LLM semantic extraction.

This module provides a bounded LLM extraction step that converts raw user
prompts into typed SemanticIntent objects. The LLM is constrained to output
only schema-validated JSON using the semantic models defined in
semantic_models.py.

The LLM never generates:
- Internal agent names or function signatures
- Execution plans or step sequences
- Arbitrary Python code
- Queue routing or tool calls

It only produces structured semantic output that downstream deterministic
code validates, grounds, and compiles.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.services.semantic_models import (
    FilterPredicate,
    OutputRequirement,
    RelationOperator,
    SemanticAmbiguity,
    SemanticConstraint,
    SemanticGoal,
    SemanticIntent,
    SemanticOperation,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
    UnsupportedRequirement,
)

logger = logging.getLogger(__name__)

SEMANTIC_EXTRACTOR_VERSION = "1.0"
SEMANTIC_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Extraction prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a structured data-operation intent extractor. Your job is to parse a user's \
natural-language instruction about data manipulation and return a JSON object that \
describes their intent using ONLY the supported operations and relation operators below.

IMPORTANT RULES:
- Preserve EVERY material user requirement. Do not silently omit anything.
- Separate multiple requested operations into distinct tasks.
- Preserve ordering and dependencies between tasks.
- Mark ambiguity instead of guessing. If something is unclear, add it to ambiguities.
- Mark unsupported requirements explicitly in unsupported_requirements.
- NEVER choose internal executors, agents, or generate arbitrary code.
- NEVER invent columns that are not in the provided column list.
- If the user requests a column that doesn't exist in the provided list, still include \
it as a column_reference with the user's exact term — grounding happens later.
- Return ONLY valid JSON matching the schema. No explanations or markdown.

SUPPORTED SEMANTIC OPERATIONS:
- clean: General data cleaning (trimming, deduplication, normalization)
- select_columns: Keep only specific columns (positive projection)
- exclude_columns: Remove specific columns (negative projection)
- filter: Filter rows based on conditions
- compare: Compare values across columns
- group: Group rows by column values
- aggregate: Compute aggregations (sum, average, count, etc.)
- derive_column: Create a new computed column
- sort: Order rows by column values
- join: Combine with another dataset
- format: Format output in a specific way
- visualize: Create charts or visualizations
- export: Export data in specific format
- limit: Limit the number of rows returned
- rename_columns: Rename one or more columns
- deduplicate: Remove duplicate rows

SUPPORTED RELATION OPERATORS (for filter predicates):
- equals: exact match
- not_equals: not equal
- greater_than: >
- greater_than_or_equal: >=
- less_than: <
- less_than_or_equal: <=
- between: value in range [min, max]
- in: value is one of a set
- not_in: value is not in a set
- contains: text contains substring
- not_contains: text does not contain substring
- matches: regex or pattern match
- is_null: value is missing/null
- is_not_null: value is present

CRITICAL DISTINCTIONS:
- "return all columns except X" → operation: exclude_columns (NOT a filter)
- "show only X and Y" → operation: select_columns (NOT a filter)
- "where X equals Y" → operation: filter
- "remove rows where X" → operation: filter (with the condition)
- "remove column X" / "hide X" / "except X" → operation: exclude_columns
- "except" in "do X except when Y" → this is a CONDITIONAL, not column exclusion

OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "goals": [{"description": "...", "priority": 1}],
  "tasks": [
    {
      "task_id": "task_1",
      "operation": {"type": "<semantic_operation>"},
      "inputs": [{"kind": "column_reference", "user_term": "..."}],
      "parameters": {},
      "depends_on": [],
      "confidence": 0.95
    }
  ],
  "outputs": [{"format": "xlsx", "description": "..."}],
  "constraints": [],
  "ambiguities": [],
  "unsupported_requirements": []
}

For filter operations, put the predicate in parameters:
{
  "task_id": "task_2",
  "operation": {"type": "filter"},
  "inputs": [{"kind": "column_reference", "user_term": "age"}],
  "parameters": {
    "predicate": {
      "left": {"kind": "column_reference", "user_term": "age"},
      "operator": "between",
      "right": {"kind": "range_value", "minimum": 18, "maximum": 25}
    }
  }
}

For exclude_columns:
{
  "task_id": "task_1",
  "operation": {"type": "exclude_columns"},
  "inputs": [{"kind": "column_reference", "user_term": "consumer ID"}]
}

For select_columns:
{
  "task_id": "task_1",
  "operation": {"type": "select_columns"},
  "inputs": [{"kind": "column_reference", "user_term": "age"}, {"kind": "column_reference", "user_term": "gender"}]
}
"""


def build_extraction_prompt(
    raw_prompt: str,
    available_columns: list[str],
    column_types: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build the messages array for the constrained LLM extraction call.

    The user message includes:
    - The raw user prompt
    - Available dataset columns with types
    - A reminder about schema constraints
    """
    column_info = []
    for col in available_columns:
        dtype = (column_types or {}).get(col, "unknown")
        column_info.append(f"  - {col} ({dtype})")

    columns_block = "\n".join(column_info) if column_info else "  (no columns available yet)"

    user_message = (
        f"USER INSTRUCTION:\n{raw_prompt}\n\n"
        f"AVAILABLE DATASET COLUMNS:\n{columns_block}\n\n"
        f"SCHEMA VERSION: {SEMANTIC_SCHEMA_VERSION}\n\n"
        "REMINDER: Do not invent columns not in the list above. "
        "Use the user's exact wording as user_term for column references. "
        "Preserve every material requirement from the instruction. "
        "Return ONLY valid JSON."
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# LLM call and response parsing
# ---------------------------------------------------------------------------


def parse_llm_semantic_response(raw_json: str | dict[str, Any]) -> SemanticIntent:
    """Parse and validate LLM output into a SemanticIntent.

    Raises ValueError if the response is not valid.
    """
    if isinstance(raw_json, str):
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
    else:
        data = raw_json

    if not isinstance(data, dict):
        raise ValueError(f"LLM returned non-object JSON: {type(data).__name__}")

    try:
        intent = SemanticIntent.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"LLM response failed schema validation: {e}") from e

    return intent


async def extract_semantic_intent(
    raw_prompt: str,
    available_columns: list[str],
    column_types: dict[str, str] | None = None,
    *,
    llm_call: Any = None,
) -> SemanticIntent:
    """Run constrained LLM semantic extraction.

    Parameters
    ----------
    raw_prompt : str
        The raw user instruction.
    available_columns : list[str]
        Column names from the uploaded dataset.
    column_types : dict[str, str] | None
        Optional mapping of column name to detected dtype.
    llm_call : callable | None
        An async callable that takes messages and returns a JSON string/dict.
        If None, falls back to the Groq integration.

    Returns
    -------
    SemanticIntent
        The validated semantic extraction result.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed or validated.
    """
    messages = build_extraction_prompt(raw_prompt, available_columns, column_types)

    if llm_call is not None:
        raw_response = await llm_call(messages)
    else:
        raw_response = await _call_groq_json(messages)

    return parse_llm_semantic_response(raw_response)


async def _call_groq_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call Groq LLM with JSON mode for structured output.

    This is the default LLM backend. Can be swapped via the llm_call
    parameter in extract_semantic_intent().
    """
    try:
        from groq import AsyncGroq
    except ImportError:
        logger.warning("groq package not installed; returning empty semantic intent")
        return {"goals": [], "tasks": [], "outputs": [], "constraints": [], "ambiguities": [], "unsupported_requirements": []}

    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set; returning empty semantic intent")
        return {"goals": [], "tasks": [], "outputs": [], "constraints": [], "ambiguities": [], "unsupported_requirements": []}

    client = AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.error("Groq semantic extraction failed: %s", e)
        raise ValueError(f"LLM extraction call failed: {e}") from e


# ---------------------------------------------------------------------------
# Synchronous extraction (for non-async contexts)
# ---------------------------------------------------------------------------


def extract_semantic_intent_sync(
    raw_prompt: str,
    available_columns: list[str],
    column_types: dict[str, str] | None = None,
    *,
    llm_call: Any = None,
) -> SemanticIntent:
    """Synchronous wrapper for extract_semantic_intent.

    Uses the provided llm_call (which must be synchronous) or falls back to
    returning an empty intent if no LLM is available.
    """
    messages = build_extraction_prompt(raw_prompt, available_columns, column_types)

    if llm_call is not None:
        raw_response = llm_call(messages)
    else:
        # Synchronous fallback: try to call Groq synchronously
        raw_response = _call_groq_json_sync(messages)

    return parse_llm_semantic_response(raw_response)


def _call_groq_json_sync(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Synchronous Groq call for non-async code paths."""
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq package not installed; returning empty semantic intent")
        return {"goals": [], "tasks": [], "outputs": [], "constraints": [], "ambiguities": [], "unsupported_requirements": []}

    import os
    import time

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set; returning empty semantic intent")
        return {"goals": [], "tasks": [], "outputs": [], "constraints": [], "ambiguities": [], "unsupported_requirements": []}

    client = Groq(api_key=api_key)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str or "429" in error_str:
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 5  # 5s, 10s
                    logger.info("Rate limited, retrying in %ds (attempt %d/%d)", wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
            logger.error("Groq semantic extraction failed: %s", e)
            raise ValueError(f"LLM extraction call failed: {e}") from e
