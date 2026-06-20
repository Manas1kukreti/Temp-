"""Schema and column grounding for semantic intent.

After semantic extraction produces a SemanticIntent with user-facing terms,
this module resolves those terms against the actual dataset schema.

Grounding rules:
- High-confidence unique match → resolve
- Multiple plausible matches → needs_clarification
- No valid match → needs_clarification
- Never silently remove unresolved references
- Never invent a column
- Preserve grounding evidence and candidates
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

from app.services.semantic_models import (
    ColumnGroundingResult,
    GroundedSemanticIntent,
    SemanticIntent,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
)

logger = logging.getLogger(__name__)

SEMANTIC_GROUNDING_VERSION = "1.0"

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
ACCEPTABLE_THRESHOLD = 0.70


def ground_semantic_intent(
    intent: SemanticIntent,
    source_columns: list[str],
    column_types: dict[str, str] | None = None,
) -> GroundedSemanticIntent:
    """Ground all column references in the semantic intent against the dataset schema.

    Returns a GroundedSemanticIntent with resolution results for each reference.
    """
    column_types = column_types or {}
    grounding_results: list[ColumnGroundingResult] = []
    unresolved: list[str] = []

    # Collect all user_terms that need grounding
    all_references = _collect_column_references(intent)

    # Ground each unique reference
    seen_terms: set[str] = set()
    for user_term in all_references:
        term_lower = user_term.strip().lower()
        if term_lower in seen_terms:
            continue
        seen_terms.add(term_lower)

        result = _ground_single_reference(user_term, source_columns)
        grounding_results.append(result)
        if result.needs_clarification or result.resolved_column is None:
            unresolved.append(user_term)

    all_resolved = len(unresolved) == 0

    return GroundedSemanticIntent(
        intent=intent,
        grounding_results=grounding_results,
        all_resolved=all_resolved,
        unresolved_references=unresolved,
    )


def _collect_column_references(intent: SemanticIntent) -> list[str]:
    """Extract all column reference user_terms from the intent."""
    terms: list[str] = []

    for task in intent.tasks:
        for inp in task.inputs:
            if inp.kind == "column_reference" and inp.user_term:
                terms.append(inp.user_term)

        # Check parameters for predicates
        predicate = task.parameters.get("predicate")
        if predicate and isinstance(predicate, dict):
            _collect_predicate_terms(predicate, terms)

        predicates = task.parameters.get("predicates")
        if predicates and isinstance(predicates, list):
            for p in predicates:
                if isinstance(p, dict):
                    _collect_predicate_terms(p, terms)

    for output in intent.outputs:
        for col_ref in output.columns:
            if col_ref.kind == "column_reference" and col_ref.user_term:
                terms.append(col_ref.user_term)

    return terms


def _collect_predicate_terms(predicate: dict[str, Any], terms: list[str]) -> None:
    """Recursively collect column reference terms from a predicate."""
    left = predicate.get("left")
    if isinstance(left, dict) and left.get("kind") == "column_reference":
        user_term = left.get("user_term", "")
        if user_term:
            terms.append(user_term)

    right = predicate.get("right")
    if isinstance(right, dict) and right.get("kind") == "column_reference":
        user_term = right.get("user_term", "")
        if user_term:
            terms.append(user_term)


def _ground_single_reference(
    user_term: str,
    source_columns: list[str],
) -> ColumnGroundingResult:
    """Ground a single user term against available columns."""
    if not source_columns:
        return ColumnGroundingResult(
            user_term=user_term,
            resolved_column=None,
            confidence=0.0,
            resolution_type="no_match",
            needs_clarification=True,
        )

    user_lower = user_term.strip().lower()
    user_normalized = _normalize_column_name(user_term)

    # Tier 1: Exact match (case-insensitive)
    for col in source_columns:
        if col.lower() == user_lower:
            return ColumnGroundingResult(
                user_term=user_term,
                resolved_column=col,
                confidence=1.0,
                resolution_type="exact_match",
            )

    # Tier 2: Case-insensitive match after stripping
    for col in source_columns:
        if col.strip().lower() == user_lower:
            return ColumnGroundingResult(
                user_term=user_term,
                resolved_column=col,
                confidence=0.99,
                resolution_type="case_insensitive_match",
            )

    # Tier 3: Normalized match (underscores, spaces, etc.)
    for col in source_columns:
        col_normalized = _normalize_column_name(col)
        if col_normalized == user_normalized:
            return ColumnGroundingResult(
                user_term=user_term,
                resolved_column=col,
                confidence=0.95,
                resolution_type="normalized_match",
            )

    # Tier 4: Semantic matching (common synonyms, abbreviations)
    semantic_match = _semantic_column_match(user_term, source_columns)
    if semantic_match:
        return semantic_match

    # Tier 5: Fuzzy match
    fuzzy_match = _fuzzy_column_match(user_term, source_columns)
    if fuzzy_match:
        return fuzzy_match

    # No match found
    # Return candidates for clarification
    candidates = _get_top_candidates(user_term, source_columns, top_n=3)
    return ColumnGroundingResult(
        user_term=user_term,
        resolved_column=None,
        confidence=0.0,
        resolution_type="no_match",
        candidates=candidates,
        needs_clarification=True,
    )


def _normalize_column_name(name: str) -> str:
    """Normalize a column name for comparison."""
    text = name.strip().lower()
    # Replace common separators with space
    text = re.sub(r"[_\-./]+", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Common abbreviation expansions
    text = re.sub(r"\bid\b", "identifier", text)
    text = re.sub(r"\bno\b", "number", text)
    text = re.sub(r"\bnum\b", "number", text)
    text = re.sub(r"\bamt\b", "amount", text)
    text = re.sub(r"\bqty\b", "quantity", text)
    text = re.sub(r"\bdesc\b", "description", text)
    return text


def _semantic_column_match(
    user_term: str,
    source_columns: list[str],
) -> ColumnGroundingResult | None:
    """Try semantic matching using known synonyms and abbreviations."""
    user_lower = user_term.strip().lower()
    user_normalized = _normalize_column_name(user_term)

    # Known semantic synonym groups
    synonym_groups: list[set[str]] = [
        {"consumer id", "customer id", "client id", "user id", "consumer identifier", "customer identifier"},
        {"gender", "sex"},
        {"age", "years old", "age years"},
        {"name", "full name", "fullname", "customer name", "consumer name"},
        {"email", "email address", "e-mail", "email id"},
        {"phone", "phone number", "telephone", "mobile", "contact number"},
        {"address", "street address", "mailing address", "postal address"},
        {"date", "transaction date", "created date", "creation date"},
        {"amount", "value", "total", "payment", "price", "cost"},
        {"status", "state", "payment status", "order status"},
        {"merchant", "vendor", "provider", "seller", "payment method"},
        {"transaction id", "txn id", "order id", "invoice id"},
        {"marital status", "relationship status"},
        {"education", "education level", "qualification", "degree"},
        {"income", "salary", "earnings", "annual income"},
        {"occupation", "job", "profession", "employment"},
    ]

    # Check if user_term is in any synonym group
    user_group: set[str] | None = None
    for group in synonym_groups:
        if user_lower in group or user_normalized in group:
            user_group = group
            break

    if user_group is None:
        return None

    # Check if any source column matches the same group
    for col in source_columns:
        col_lower = col.strip().lower()
        col_normalized = _normalize_column_name(col)
        if col_lower in user_group or col_normalized in user_group:
            return ColumnGroundingResult(
                user_term=user_term,
                resolved_column=col,
                confidence=0.88,
                resolution_type="semantic_column_match",
            )

    # Partial match: check if normalized column contains any group term
    for col in source_columns:
        col_normalized = _normalize_column_name(col)
        for synonym in user_group:
            if synonym in col_normalized or col_normalized in synonym:
                return ColumnGroundingResult(
                    user_term=user_term,
                    resolved_column=col,
                    confidence=0.82,
                    resolution_type="semantic_column_match",
                    candidates=[col],
                )

    return None


def _fuzzy_column_match(
    user_term: str,
    source_columns: list[str],
) -> ColumnGroundingResult | None:
    """Use fuzzy string matching as a last resort."""
    user_normalized = _normalize_column_name(user_term)

    best_col: str | None = None
    best_score: float = 0.0

    for col in source_columns:
        col_normalized = _normalize_column_name(col)

        # SequenceMatcher ratio
        score = difflib.SequenceMatcher(None, user_normalized, col_normalized).ratio()

        # Token overlap bonus
        token_score = _token_overlap_score(user_normalized, col_normalized)
        score = max(score, token_score)

        if score > best_score:
            best_score = score
            best_col = col

    if best_col and best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return ColumnGroundingResult(
            user_term=user_term,
            resolved_column=best_col,
            confidence=best_score,
            resolution_type="fuzzy_match",
        )

    if best_col and best_score >= ACCEPTABLE_THRESHOLD:
        # Lower confidence — resolve but flag
        return ColumnGroundingResult(
            user_term=user_term,
            resolved_column=best_col,
            confidence=best_score,
            resolution_type="fuzzy_match",
            candidates=_get_top_candidates(user_term, source_columns, top_n=3),
        )

    return None


def _token_overlap_score(left: str, right: str) -> float:
    """Compute token overlap ratio between two normalized strings."""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _get_top_candidates(user_term: str, source_columns: list[str], top_n: int = 3) -> list[str]:
    """Return the top N closest columns by similarity."""
    user_normalized = _normalize_column_name(user_term)
    scored = []
    for col in source_columns:
        col_normalized = _normalize_column_name(col)
        score = difflib.SequenceMatcher(None, user_normalized, col_normalized).ratio()
        scored.append((col, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in scored[:top_n]]
