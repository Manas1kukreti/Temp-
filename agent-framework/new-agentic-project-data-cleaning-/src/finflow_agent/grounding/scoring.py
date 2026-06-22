"""Deterministic scoring utilities for the Candidate Generation Layer.

Provides pure scoring functions used by both Column Grounder and Predicate Grounder
to produce scored candidate columns from semantic profiles.

Scoring dimensions:
- Token overlap (Jaccard similarity of tokenized names)
- Value-concept matching (reference text vs. actual column values)
- Semantic-type alignment (reference kind vs. column role compatibility)
- Column-name similarity (normalized Levenshtein distance)

All functions are pure/deterministic: same inputs always produce same output.
No randomness, no external state, no side effects.

Requirements: 7.2
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Regex to split on camelCase boundaries: inserts a boundary before uppercase
# letters that follow lowercase letters or before uppercase letters followed
# by lowercase when preceded by another uppercase (e.g., "XMLParser" -> "XML", "Parser")
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z])(?=[A-Z])"  # lowercase followed by uppercase
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # uppercase followed by uppercase+lowercase
)


def tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on underscores, spaces, and camelCase boundaries.

    All tokens are lowercased and empty tokens are removed.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens with empty strings removed.

    Examples:
        >>> tokenize("payment_method")
        ['payment', 'method']
        >>> tokenize("paymentMethod")
        ['payment', 'method']
        >>> tokenize("XMLParser")
        ['xml', 'parser']
        >>> tokenize("total amount")
        ['total', 'amount']
        >>> tokenize("")
        []
    """
    # Step 1: Split on camelCase boundaries
    parts = _CAMEL_BOUNDARY.split(text)

    # Step 2: Further split each part on underscores and spaces
    tokens: list[str] = []
    for part in parts:
        sub_parts = re.split(r"[_\s]+", part)
        for sub in sub_parts:
            lowered = sub.lower()
            if lowered:  # Remove empty tokens
                tokens.append(lowered)

    return tokens


# ---------------------------------------------------------------------------
# Token Overlap Scoring (Jaccard Similarity)
# ---------------------------------------------------------------------------


def compute_token_overlap(reference_text: str, column_name: str) -> float:
    """Compute Jaccard similarity of token sets from reference text and column name.

    Tokenizes both strings (split on underscore, camelCase boundaries, spaces)
    and computes the Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|.

    Args:
        reference_text: The semantic reference text from the user prompt.
        column_name: The physical column name from the dataset.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if both token sets are empty.
        1.0 means identical token sets.

    Examples:
        >>> compute_token_overlap("payment method", "payment_method")
        1.0
        >>> compute_token_overlap("amount", "total_amount")
        0.5
        >>> compute_token_overlap("", "")
        0.0
    """
    ref_tokens = set(tokenize(reference_text))
    col_tokens = set(tokenize(column_name))

    if not ref_tokens and not col_tokens:
        return 0.0

    union = ref_tokens | col_tokens
    if not union:
        return 0.0

    intersection = ref_tokens & col_tokens
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Value-Concept Matching
# ---------------------------------------------------------------------------


def compute_value_concept_match(
    reference_text: str, column_values: list[str]
) -> float:
    """Check if reference text or its tokens appear as values in the column.

    Scoring logic:
    - Exact match of reference_text in column values → 1.0
    - All reference tokens found in at least one value → 0.8
    - Partial token overlap with column values → proportional score (0.0–0.6)
    - No match → 0.0

    Args:
        reference_text: The semantic reference text from the user prompt.
        column_values: List of string values present in the column.

    Returns:
        Float in [0.0, 1.0]. Higher score if reference matches actual column values.

    Examples:
        >>> compute_value_concept_match("paypal", ["paypal", "cash", "credit"])
        1.0
        >>> compute_value_concept_match("unknown", ["paypal", "cash"])
        0.0
    """
    if not reference_text or not column_values:
        return 0.0

    ref_lower = reference_text.lower().strip()

    # Check exact match (case-insensitive)
    values_lower = [v.lower().strip() for v in column_values]
    if ref_lower in values_lower:
        return 1.0

    # Tokenize the reference text
    ref_tokens = set(tokenize(reference_text))
    if not ref_tokens:
        return 0.0

    # Check if all tokens appear in any single value
    for val in values_lower:
        val_tokens = set(tokenize(val))
        if ref_tokens <= val_tokens:  # All ref tokens found in this value
            return 0.8

    # Check partial token overlap across all values
    all_value_tokens: set[str] = set()
    for val in values_lower:
        all_value_tokens.update(tokenize(val))

    if not all_value_tokens:
        return 0.0

    matched_tokens = ref_tokens & all_value_tokens
    if not matched_tokens:
        return 0.0

    # Proportional partial match, capped at 0.6
    overlap_ratio = len(matched_tokens) / len(ref_tokens)
    return min(overlap_ratio * 0.6, 0.6)


# ---------------------------------------------------------------------------
# Semantic-Type Alignment Scoring
# ---------------------------------------------------------------------------

# Mapping of reference_kind to expected column_role compatibility scores.
# Higher scores indicate better alignment between the reference type and
# what kind of column role would be expected.
_SEMANTIC_TYPE_ALIGNMENT: dict[str, dict[str, float]] = {
    "explicit_name": {
        # Explicit names can match any role — baseline score
        "_default": 0.5,
    },
    "semantic_concept": {
        # Semantic concepts align well with measure, dimension, categorical
        "measure": 0.7,
        "dimension": 0.7,
        "categorical": 0.7,
        "_default": 0.4,
    },
    "value_implied": {
        # Value-implied references align best with categorical and dimension
        "categorical": 0.8,
        "dimension": 0.8,
        "_default": 0.3,
    },
    "generic_reference": {
        # Generic references (e.g., "field", "column") — low baseline
        "_default": 0.3,
    },
    "column_group": {
        # Column groups can match dimension or categorical groupings
        "dimension": 0.6,
        "categorical": 0.6,
        "_default": 0.4,
    },
}


def compute_semantic_type_alignment(
    reference_kind: str, column_role: str
) -> float:
    """Score alignment between a reference kind and a column role.

    Maps reference kinds to expected column roles:
    - explicit_name → any role (0.5 baseline)
    - semantic_concept → measure, dimension, categorical (0.7)
    - value_implied → categorical, dimension (0.8 if match)
    - generic_reference → any (0.3 baseline)
    - column_group → dimension, categorical (0.6)

    Args:
        reference_kind: The classified reference kind (e.g., "explicit_name",
            "semantic_concept", "value_implied", "generic_reference", "column_group").
        column_role: The inferred column role (e.g., "measure", "dimension",
            "categorical", "identifier", "temporal", "text", "boolean", "unknown").

    Returns:
        Float in [0.0, 1.0].

    Examples:
        >>> compute_semantic_type_alignment("semantic_concept", "measure")
        0.7
        >>> compute_semantic_type_alignment("value_implied", "categorical")
        0.8
        >>> compute_semantic_type_alignment("generic_reference", "text")
        0.3
    """
    kind_lower = reference_kind.lower().strip()
    role_lower = column_role.lower().strip()

    kind_map = _SEMANTIC_TYPE_ALIGNMENT.get(kind_lower)
    if kind_map is None:
        # Unknown reference kind → low score
        return 0.2

    return kind_map.get(role_lower, kind_map.get("_default", 0.3))


# ---------------------------------------------------------------------------
# Column-Name Similarity (Normalized Levenshtein)
# ---------------------------------------------------------------------------


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses the standard dynamic programming approach.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer edit distance (number of insertions, deletions, substitutions).
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            current_row.append(
                min(
                    current_row[j] + 1,       # insertion
                    previous_row[j + 1] + 1,  # deletion
                    previous_row[j] + cost,   # substitution
                )
            )
        previous_row = current_row

    return previous_row[-1]


def compute_name_similarity(reference_text: str, column_name: str) -> float:
    """Compute normalized Levenshtein distance-based similarity.

    Case-insensitive comparison. Returns 1.0 for exact match, 0.0 for
    completely different strings.

    Formula: 1.0 - (edit_distance / max(len(s1), len(s2)))

    Args:
        reference_text: The semantic reference text from the user prompt.
        column_name: The physical column name from the dataset.

    Returns:
        Float in [0.0, 1.0] where 1.0 is exact match (case-insensitive).

    Examples:
        >>> compute_name_similarity("payment_method", "payment_method")
        1.0
        >>> compute_name_similarity("Payment_Method", "payment_method")
        1.0
        >>> compute_name_similarity("abc", "xyz")
        0.0
    """
    s1 = reference_text.lower().strip()
    s2 = column_name.lower().strip()

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    distance = _levenshtein_distance(s1, s2)

    return 1.0 - (distance / max_len)
