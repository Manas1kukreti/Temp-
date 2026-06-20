"""Semantic coverage verification.

After extraction and before compilation, verifies that every material
requirement from the raw prompt is represented in the semantic intent.

This catches cases like:
- "clean this data and return all columns except consumer ID"
  → extraction only produced clean task, missed exclude_columns

The coverage checker uses a bounded LLM call to compare the original
prompt against the extracted intent and identify gaps.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.services.semantic_models import (
    CoverageResult,
    MissingRequirement,
    ConflictingRequirement,
    SemanticIntent,
    SemanticOperationType,
)

logger = logging.getLogger(__name__)

COVERAGE_CHECKER_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Deterministic coverage checks (fast path)
# ---------------------------------------------------------------------------


def check_coverage_deterministic(
    raw_prompt: str,
    intent: SemanticIntent,
) -> CoverageResult:
    """Fast deterministic coverage check using keyword analysis.

    This catches obvious omissions without an LLM call:
    - Prompt mentions column exclusion but no exclude_columns task exists
    - Prompt mentions filtering but no filter task exists
    - Prompt mentions cleaning but no clean task exists
    - Prompt mentions sorting but no sort task exists
    - etc.
    """
    prompt_lower = raw_prompt.strip().lower()
    missing: list[MissingRequirement] = []
    conflicts: list[ConflictingRequirement] = []

    task_types = {t.operation.type for t in intent.tasks}

    # Check for exclusion intent
    if _prompt_has_exclusion_intent(prompt_lower):
        if SemanticOperationType.exclude_columns not in task_types:
            # Could also be a filter exclusion — check if it's about rows
            if not _is_row_exclusion(prompt_lower):
                missing.append(MissingRequirement(
                    description="Prompt indicates column exclusion but no exclude_columns task was extracted.",
                    source_text=_extract_exclusion_fragment(prompt_lower),
                    suggested_operation=SemanticOperationType.exclude_columns,
                ))

    # Check for positive projection intent
    # BUT: if the prompt has filter intent (e.g. "return only those WHERE..."),
    # the word "only" refers to ROW filtering, not column selection.
    if _prompt_has_selection_intent(prompt_lower):
        if (SemanticOperationType.select_columns not in task_types
                and SemanticOperationType.exclude_columns not in task_types):
            # Don't flag if this is actually a filter expression
            # ("return only those which have..." is a filter, not column selection)
            if not _prompt_has_filter_intent(prompt_lower) and SemanticOperationType.filter not in task_types:
                missing.append(MissingRequirement(
                    description="Prompt indicates column selection but no select_columns or exclude_columns task was extracted.",
                    source_text=_extract_selection_fragment(prompt_lower),
                    suggested_operation=SemanticOperationType.select_columns,
                ))

    # Check for filter intent
    if _prompt_has_filter_intent(prompt_lower):
        if SemanticOperationType.filter not in task_types:
            missing.append(MissingRequirement(
                description="Prompt indicates row filtering but no filter task was extracted.",
                source_text="",
                suggested_operation=SemanticOperationType.filter,
            ))

    # Check for clean intent
    if _prompt_has_clean_intent(prompt_lower):
        if SemanticOperationType.clean not in task_types and SemanticOperationType.deduplicate not in task_types:
            missing.append(MissingRequirement(
                description="Prompt indicates data cleaning but no clean task was extracted.",
                source_text="",
                suggested_operation=SemanticOperationType.clean,
            ))

    # Check for sort intent
    if _prompt_has_sort_intent(prompt_lower):
        if SemanticOperationType.sort not in task_types:
            missing.append(MissingRequirement(
                description="Prompt indicates sorting but no sort task was extracted.",
                source_text="",
                suggested_operation=SemanticOperationType.sort,
            ))

    # Check for limit intent
    if _prompt_has_limit_intent(prompt_lower):
        if SemanticOperationType.limit not in task_types:
            missing.append(MissingRequirement(
                description="Prompt indicates row limiting but no limit task was extracted.",
                source_text="",
                suggested_operation=SemanticOperationType.limit,
            ))

    # Check for conflicts: both select and exclude columns
    if (SemanticOperationType.select_columns in task_types
            and SemanticOperationType.exclude_columns in task_types):
        conflicts.append(ConflictingRequirement(
            description="Both select_columns and exclude_columns tasks exist — these may conflict.",
        ))

    covered = len(missing) == 0
    return CoverageResult(
        covered=covered,
        missing_requirements=missing,
        conflicting_requirements=conflicts,
    )


# ---------------------------------------------------------------------------
# LLM-assisted coverage verification (for complex prompts)
# ---------------------------------------------------------------------------


_COVERAGE_SYSTEM_PROMPT = """\
You are a coverage verifier. Given a user's original instruction and the list of \
operations that were extracted from it, determine if ANY material requirement from \
the instruction is MISSING from the extraction.

A "material requirement" is any action the user wants performed on their data. \
Ignore filler words, politeness, or commentary.

Return a JSON object:
{
  "covered": true/false,
  "missing_requirements": [
    {
      "description": "what was missed",
      "source_text": "the exact phrase from the instruction",
      "suggested_operation": "exclude_columns|select_columns|filter|clean|sort|limit|..."
    }
  ],
  "conflicting_requirements": []
}

Return {"covered": true, "missing_requirements": [], "conflicting_requirements": []} \
if everything is accounted for.
"""


def build_coverage_check_prompt(
    raw_prompt: str,
    intent: SemanticIntent,
) -> list[dict[str, str]]:
    """Build messages for the LLM coverage check."""
    task_descriptions = []
    for task in intent.tasks:
        inputs_str = ", ".join(i.user_term for i in task.inputs if i.user_term)
        task_descriptions.append(
            f"  - {task.operation.type.value}: {inputs_str or '(no specific columns)'}"
        )

    ambiguity_notes = []
    for amb in intent.ambiguities:
        ambiguity_notes.append(f"  - AMBIGUOUS: {amb.description}")

    unsupported_notes = []
    for uns in intent.unsupported_requirements:
        unsupported_notes.append(f"  - UNSUPPORTED: {uns.description}")

    tasks_block = "\n".join(task_descriptions) if task_descriptions else "  (no tasks extracted)"
    extras = "\n".join(ambiguity_notes + unsupported_notes)

    user_msg = (
        f"ORIGINAL USER INSTRUCTION:\n{raw_prompt}\n\n"
        f"EXTRACTED OPERATIONS:\n{tasks_block}\n"
    )
    if extras:
        user_msg += f"\nADDITIONAL NOTES:\n{extras}\n"
    user_msg += "\nIs every material requirement from the instruction represented above?"

    return [
        {"role": "system", "content": _COVERAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


async def check_coverage_with_llm(
    raw_prompt: str,
    intent: SemanticIntent,
    *,
    llm_call: Any = None,
) -> CoverageResult:
    """Use LLM to verify coverage of the extraction.

    This is the deep check — used when the deterministic check passes but
    we want extra confidence for complex prompts.
    """
    messages = build_coverage_check_prompt(raw_prompt, intent)

    if llm_call is not None:
        raw_response = await llm_call(messages)
    else:
        raw_response = await _call_groq_coverage(messages)

    return _parse_coverage_response(raw_response)


def _parse_coverage_response(raw: str | dict[str, Any]) -> CoverageResult:
    """Parse the LLM coverage check response."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Coverage LLM returned invalid JSON; assuming uncovered")
            return CoverageResult(
                covered=False,
                missing_requirements=[MissingRequirement(
                    description="Coverage check failed to parse LLM response",
                )],
            )
    else:
        data = raw

    if not isinstance(data, dict):
        return CoverageResult(covered=False, missing_requirements=[
            MissingRequirement(description="Coverage check returned non-dict")
        ])

    covered = bool(data.get("covered", False))
    missing = []
    for item in data.get("missing_requirements", []):
        if isinstance(item, dict):
            suggested_op = None
            op_str = item.get("suggested_operation", "")
            if op_str:
                try:
                    suggested_op = SemanticOperationType(op_str)
                except ValueError:
                    pass
            missing.append(MissingRequirement(
                description=str(item.get("description", "")),
                source_text=str(item.get("source_text", "")),
                suggested_operation=suggested_op,
            ))

    conflicts = []
    for item in data.get("conflicting_requirements", []):
        if isinstance(item, dict):
            conflicts.append(ConflictingRequirement(
                description=str(item.get("description", "")),
            ))

    return CoverageResult(
        covered=covered,
        missing_requirements=missing,
        conflicting_requirements=conflicts,
    )


async def _call_groq_coverage(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call Groq for coverage verification."""
    try:
        from groq import AsyncGroq
    except ImportError:
        return {"covered": True, "missing_requirements": [], "conflicting_requirements": []}

    import os
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"covered": True, "missing_requirements": [], "conflicting_requirements": []}

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
        logger.error("Coverage verification LLM call failed: %s", e)
        return {"covered": True, "missing_requirements": [], "conflicting_requirements": []}


# ---------------------------------------------------------------------------
# Prompt analysis helpers (deterministic)
# ---------------------------------------------------------------------------


def _prompt_has_exclusion_intent(prompt_lower: str) -> bool:
    """Detect column exclusion intent in a prompt."""
    import re
    exclusion_patterns = [
        r"\bexcept\b",
        r"\bexclude\b",
        r"\bwithout\b",
        r"\bhide\b",
        r"\bremove\b.*\bcolumn",
        r"\bdrop\b.*\bcolumn",
        r"\bdo not include\b",
        r"\bdon'?t include\b",
        r"\bshould not appear\b",
        r"\beverything but\b",
        r"\ball (?:columns |fields )?(?:other than|besides)\b",
        r"\bnot (?:show|return|include|display)\b",
        r"\bother than\b",
    ]
    return any(re.search(p, prompt_lower) for p in exclusion_patterns)


def _is_row_exclusion(prompt_lower: str) -> bool:
    """Distinguish row exclusion from column exclusion."""
    import re
    row_patterns = [
        r"\bexclude\b.*\brows?\b",
        r"\bremove\b.*\brows?\b",
        r"\bdrop\b.*\brows?\b",
        r"\brows?\b.*\bexcept\b",
        r"\bexcept when\b",
        r"\bexcept for\b.*\btransaction",
        r"\bexcept those\b",
    ]
    return any(re.search(p, prompt_lower) for p in row_patterns)


def _extract_exclusion_fragment(prompt_lower: str) -> str:
    """Extract the exclusion-related fragment from the prompt."""
    import re
    patterns = [
        r"((?:all columns |everything |all fields )?except\b.{1,60})",
        r"((?:hide|exclude|without|remove)\b.{1,60})",
        r"(do not include\b.{1,60})",
        r"(should not appear\b.{1,30})",
    ]
    for p in patterns:
        match = re.search(p, prompt_lower)
        if match:
            return match.group(1).strip()
    return ""


def _prompt_has_selection_intent(prompt_lower: str) -> bool:
    """Detect positive column selection intent.

    Must distinguish between:
    - "show only age and gender" → column selection (YES)
    - "return only those which have credit score > 600" → row filter (NO)
    - "I only need age and gender" → column selection (YES)
    """
    import re

    # Reject filter-like "only" usage (only those, only rows, only records where...)
    if re.search(r"\bonly\s+(?:those|the ones|rows?|records?|entries?|items?)\b", prompt_lower):
        return False
    if re.search(r"\breturn\s+only\s+(?:those|the ones|rows?|records?)\b", prompt_lower):
        return False

    patterns = [
        r"\bonly\b.*\b(?:show|return|give|keep|output|extract)\b",
        r"\b(?:show|return|give|keep|output|extract)\b.*\bonly\b",
        r"\bjust\b.*\b(?:show|return|give|keep)\b",
        r"\b(?:show|return|give|keep)\b.*\bjust\b",
        r"\bonly need\b",
        r"\bjust need\b",
    ]
    return any(re.search(p, prompt_lower) for p in patterns)


def _extract_selection_fragment(prompt_lower: str) -> str:
    """Extract the selection-related fragment from the prompt."""
    import re
    match = re.search(r"((?:only|just)\b.{1,60})", prompt_lower)
    return match.group(1).strip() if match else ""


def _prompt_has_filter_intent(prompt_lower: str) -> bool:
    """Detect row filtering intent."""
    import re
    patterns = [
        r"\bwhere\b",
        r"\bfilter\b",
        r"\bequals?\b",
        r"\bgreater than\b",
        r"\bless than\b",
        r"\bbetween\b.*\band\b",
        r"\bat least\b",
        r"\bat most\b",
        r"\bcontains?\b",
        r"\brows? (?:with|for|where)\b",
    ]
    return any(re.search(p, prompt_lower) for p in patterns)


def _prompt_has_clean_intent(prompt_lower: str) -> bool:
    """Detect cleaning intent."""
    import re
    patterns = [
        r"\bclean\b",
        r"\bnormali[sz]e\b",
        r"\bstandardi[sz]e\b",
        r"\bdeduplicate\b",
        r"\bremove duplicates\b",
        r"\btrim\b",
    ]
    return any(re.search(p, prompt_lower) for p in patterns)


def _prompt_has_sort_intent(prompt_lower: str) -> bool:
    """Detect sorting intent."""
    import re
    return bool(re.search(r"\b(?:sort|order)\s+by\b", prompt_lower))


def _prompt_has_limit_intent(prompt_lower: str) -> bool:
    """Detect row limiting intent."""
    import re
    return bool(re.search(r"\b(?:top|first|limit|only first)\s+\d+\b", prompt_lower))
