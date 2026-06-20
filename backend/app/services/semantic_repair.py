"""Bounded semantic repair.

When coverage verification detects an omission in the extracted intent,
this module runs ONE bounded repair extraction to fill the gap.

Rules:
- Maximum 1 repair attempt (no infinite loops)
- Repair can only ADD or CORRECT tasks, not create execution plans
- Repaired result is re-validated against the semantic schema
- If still incomplete after repair, return needs_clarification status
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.services.semantic_models import (
    CoverageResult,
    MissingRequirement,
    SemanticIntent,
    SemanticTask,
)

logger = logging.getLogger(__name__)

SEMANTIC_REPAIR_VERSION = "1.0"


_REPAIR_SYSTEM_PROMPT = """\
You are a semantic intent repair agent. A previous extraction missed some requirements \
from the user's instruction. Your job is to produce ONLY the MISSING tasks that need \
to be added to the existing extraction.

RULES:
- Return ONLY the additional tasks to add, not the full intent.
- Use the same semantic operation types and format as the original extraction.
- Do NOT duplicate tasks that already exist.
- Do NOT generate execution plans, agent names, or code.
- Return valid JSON matching this format:

{
  "additional_tasks": [
    {
      "task_id": "repair_1",
      "operation": {"type": "<semantic_operation>"},
      "inputs": [{"kind": "column_reference", "user_term": "..."}],
      "parameters": {},
      "depends_on": [],
      "confidence": 0.85
    }
  ],
  "repair_notes": ["explanation of what was added"]
}

SUPPORTED OPERATIONS: clean, select_columns, exclude_columns, filter, compare, \
group, aggregate, derive_column, sort, join, format, visualize, export, limit, \
rename_columns, deduplicate
"""


def build_repair_prompt(
    raw_prompt: str,
    current_intent: SemanticIntent,
    missing_requirements: list[MissingRequirement],
    available_columns: list[str],
) -> list[dict[str, str]]:
    """Build the repair prompt messages."""
    # Describe current state
    current_tasks = []
    for task in current_intent.tasks:
        inputs_str = ", ".join(i.user_term for i in task.inputs if i.user_term)
        current_tasks.append(f"  - {task.task_id}: {task.operation.type.value}({inputs_str})")
    current_block = "\n".join(current_tasks) if current_tasks else "  (none)"

    # Describe what's missing
    missing_descs = []
    for req in missing_requirements:
        desc = req.description
        if req.source_text:
            desc += f' (from: "{req.source_text}")'
        if req.suggested_operation:
            desc += f" [suggested: {req.suggested_operation.value}]"
        missing_descs.append(f"  - {desc}")
    missing_block = "\n".join(missing_descs)

    columns_block = ", ".join(available_columns) if available_columns else "(none available)"

    user_msg = (
        f"ORIGINAL USER INSTRUCTION:\n{raw_prompt}\n\n"
        f"CURRENTLY EXTRACTED TASKS:\n{current_block}\n\n"
        f"MISSING REQUIREMENTS:\n{missing_block}\n\n"
        f"AVAILABLE COLUMNS: {columns_block}\n\n"
        "Please produce ONLY the additional tasks needed to cover the missing requirements."
    )

    return [
        {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


async def repair_semantic_intent(
    raw_prompt: str,
    current_intent: SemanticIntent,
    coverage_result: CoverageResult,
    available_columns: list[str],
    *,
    llm_call: Any = None,
) -> tuple[SemanticIntent, list[str]]:
    """Run one bounded repair attempt.

    Returns the repaired SemanticIntent and a list of repair notes.
    If repair fails, returns the original intent unchanged.
    """
    if coverage_result.covered:
        return current_intent, []

    messages = build_repair_prompt(
        raw_prompt,
        current_intent,
        coverage_result.missing_requirements,
        available_columns,
    )

    try:
        if llm_call is not None:
            raw_response = await llm_call(messages)
        else:
            raw_response = await _call_groq_repair(messages)

        additional_tasks, repair_notes = _parse_repair_response(raw_response)

        if not additional_tasks:
            logger.info("Repair produced no additional tasks")
            return current_intent, ["Repair attempted but produced no additional tasks."]

        # Merge additional tasks into the intent
        merged_tasks = list(current_intent.tasks) + additional_tasks
        repaired = SemanticIntent(
            goals=current_intent.goals,
            tasks=merged_tasks,
            outputs=current_intent.outputs,
            constraints=current_intent.constraints,
            ambiguities=current_intent.ambiguities,
            unsupported_requirements=current_intent.unsupported_requirements,
        )

        return repaired, repair_notes

    except Exception as e:
        logger.error("Semantic repair failed: %s", e)
        return current_intent, [f"Repair failed: {e}"]


def _parse_repair_response(raw: str | dict[str, Any]) -> tuple[list[SemanticTask], list[str]]:
    """Parse the repair LLM response."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Repair LLM returned invalid JSON: %s", e)
            return [], ["Repair response was not valid JSON."]
    else:
        data = raw

    if not isinstance(data, dict):
        return [], ["Repair response was not a dict."]

    repair_notes = data.get("repair_notes", [])
    if not isinstance(repair_notes, list):
        repair_notes = [str(repair_notes)]

    additional_tasks: list[SemanticTask] = []
    for task_data in data.get("additional_tasks", []):
        if not isinstance(task_data, dict):
            continue
        try:
            task = SemanticTask.model_validate(task_data)
            additional_tasks.append(task)
        except ValidationError as e:
            logger.warning("Repair task failed validation: %s", e)
            repair_notes.append(f"One repair task failed validation: {e}")

    return additional_tasks, repair_notes


async def _call_groq_repair(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call Groq for repair."""
    try:
        from groq import AsyncGroq
    except ImportError:
        return {"additional_tasks": [], "repair_notes": ["groq not installed"]}

    import os
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"additional_tasks": [], "repair_notes": ["GROQ_API_KEY not set"]}

    client = AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.error("Repair LLM call failed: %s", e)
        return {"additional_tasks": [], "repair_notes": [f"LLM call failed: {e}"]}
