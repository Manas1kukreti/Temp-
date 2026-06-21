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
You are a data-operation intent extractor. Parse the user instruction into structured JSON.

RULES:
- Preserve EVERY user requirement. Never silently omit.
- Separate multiple operations into distinct tasks.
- Mark ambiguity instead of guessing.
- Never invent columns not in the provided list.
- Return ONLY valid JSON. No explanations.

OPERATIONS: clean, select_columns, exclude_columns, filter, aggregate, sort, limit, deduplicate, rename_columns, derive_column, visualize, export

OPERATORS (for filter): equals, not_equals, greater_than, greater_than_or_equal, less_than, less_than_or_equal, between, in, not_in, contains, not_contains, is_null, is_not_null

KEY DISTINCTIONS:
- "except X" / "hide X" / "without X" → exclude_columns
- "only X and Y" → select_columns
- "where X > Y" / "rows with" → filter
- "except when Y" → conditional filter (NOT column exclusion)

OUTPUT JSON:
{"goals":[{"description":"...","priority":1}],"tasks":[{"task_id":"task_1","operation":{"type":"<op>"},"inputs":[{"kind":"column_reference","user_term":"..."}],"parameters":{},"depends_on":[],"confidence":0.95}],"outputs":[],"constraints":[],"ambiguities":[],"unsupported_requirements":[]}

For filter: parameters.predicate = {"left":{"kind":"column_reference","user_term":"col"},"operator":"<op>","right":{"kind":"literal_value","value":123}}
For between: right = {"kind":"range_value","minimum":18,"maximum":25}
For in: right = {"kind":"list_value","values":["a","b"]}
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

    Includes preprocessing to handle common LLM output inconsistencies:
    - ambiguities as plain strings → convert to SemanticAmbiguity objects
    - unsupported_requirements as plain strings → convert to objects
    - depends_on as malformed strings → fix to proper lists
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

    # Preprocess: fix common LLM output issues before validation
    data = _preprocess_llm_output(data)

    try:
        intent = SemanticIntent.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"LLM response failed schema validation: {e}") from e

    return intent


def _preprocess_llm_output(data: dict[str, Any]) -> dict[str, Any]:
    """Fix common LLM output inconsistencies before Pydantic validation."""
    # Fix ambiguities: convert plain strings to objects
    if "ambiguities" in data and isinstance(data["ambiguities"], list):
        fixed_ambiguities = []
        for item in data["ambiguities"]:
            if isinstance(item, str):
                fixed_ambiguities.append({
                    "description": item,
                    "possible_interpretations": [],
                    "source_text": "",
                })
            elif isinstance(item, dict):
                fixed_ambiguities.append(item)
        data["ambiguities"] = fixed_ambiguities

    # Fix unsupported_requirements: convert plain strings to objects
    if "unsupported_requirements" in data and isinstance(data["unsupported_requirements"], list):
        fixed_unsupported = []
        for item in data["unsupported_requirements"]:
            if isinstance(item, str):
                fixed_unsupported.append({
                    "description": item,
                    "reason": "",
                    "source_text": "",
                })
            elif isinstance(item, dict):
                fixed_unsupported.append(item)
        data["unsupported_requirements"] = fixed_unsupported

    # Fix constraints: convert plain strings to objects
    if "constraints" in data and isinstance(data["constraints"], list):
        fixed_constraints = []
        for item in data["constraints"]:
            if isinstance(item, str):
                fixed_constraints.append({
                    "constraint_type": "general",
                    "description": item,
                    "parameters": {},
                })
            elif isinstance(item, dict):
                fixed_constraints.append(item)
        data["constraints"] = fixed_constraints

    # Fix tasks: ensure depends_on is always a list
    if "tasks" in data and isinstance(data["tasks"], list):
        for task in data["tasks"]:
            if not isinstance(task, dict):
                continue
            # Fix depends_on if it's a string or malformed
            deps = task.get("depends_on")
            if isinstance(deps, str):
                # Try to parse as JSON list
                try:
                    task["depends_on"] = json.loads(deps)
                except (json.JSONDecodeError, TypeError):
                    task["depends_on"] = [deps] if deps.strip() else []
            elif deps is None:
                task["depends_on"] = []

            # Ensure confidence is a number
            conf = task.get("confidence")
            if isinstance(conf, str):
                try:
                    task["confidence"] = float(conf)
                except (TypeError, ValueError):
                    task["confidence"] = None

    return data


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
            # Groq sometimes rejects valid-ish JSON via json_validate_failed
            # but includes the generated output — try to salvage it
            if "failed_generation" in error_str:
                salvaged = _try_salvage_failed_generation(e)
                if salvaged is not None:
                    return salvaged
            logger.error("Groq semantic extraction failed: %s", e)
            raise ValueError(f"LLM extraction call failed: {e}") from e


def _try_salvage_failed_generation(exc: Exception) -> dict[str, Any] | None:
    """Try to extract usable JSON from Groq's failed_generation error.

    When Groq's JSON mode rejects output due to minor formatting issues,
    the actual generated text is included in the error. We attempt to parse
    it and fix common issues.
    """
    import re
    error_str = str(exc)

    # Extract the failed_generation JSON from the error message
    match = re.search(r"'failed_generation':\s*'(.*?)'}\s*$", error_str, re.DOTALL)
    if not match:
        # Try alternate format
        match = re.search(r'"failed_generation":\s*"(.*?)"\s*}', error_str, re.DOTALL)
    if not match:
        return None

    raw_json = match.group(1)
    # Unescape
    raw_json = raw_json.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

    try:
        data = json.loads(raw_json)
        if isinstance(data, dict):
            logger.info("Salvaged failed_generation JSON successfully")
            return _preprocess_llm_output(data)
    except json.JSONDecodeError:
        pass

    # Try fixing common issues: depends_on as string
    raw_json_fixed = re.sub(
        r'"depends_on=\[([^\]]*)\]"',
        lambda m: '"depends_on": [' + m.group(1) + ']',
        raw_json,
    )
    try:
        data = json.loads(raw_json_fixed)
        if isinstance(data, dict):
            logger.info("Salvaged failed_generation JSON after fixing depends_on")
            return _preprocess_llm_output(data)
    except json.JSONDecodeError:
        pass

    return None
