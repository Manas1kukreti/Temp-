"""Typed semantic intermediate representation for user intent.

This module defines the semantic layer that sits between raw user prompts
and canonical intent actions. The LLM extraction step produces instances
of these models; deterministic code then validates, grounds, and compiles
them into canonical actions.

These models describe *user meaning* without referencing internal agent
names, function signatures, queue routing, or execution mechanics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums: Supported semantic operations and relation operators
# ---------------------------------------------------------------------------


class SemanticOperationType(str, Enum):
    """Operations the system can semantically understand from user prompts.

    These do NOT map 1:1 to internal agent operations — the compiler handles
    that translation. This enum represents user-facing *meanings*.
    """

    clean = "clean"
    select_columns = "select_columns"
    exclude_columns = "exclude_columns"
    filter = "filter"
    compare = "compare"
    group = "group"
    aggregate = "aggregate"
    derive_column = "derive_column"
    sort = "sort"
    join = "join"
    format = "format"
    visualize = "visualize"
    export = "export"
    limit = "limit"
    rename_columns = "rename_columns"
    deduplicate = "deduplicate"


class RelationOperator(str, Enum):
    """Supported relation operators for filter predicates."""

    equals = "equals"
    not_equals = "not_equals"
    greater_than = "greater_than"
    greater_than_or_equal = "greater_than_or_equal"
    less_than = "less_than"
    less_than_or_equal = "less_than_or_equal"
    between = "between"
    in_ = "in"
    not_in = "not_in"
    contains = "contains"
    not_contains = "not_contains"
    matches = "matches"
    is_null = "is_null"
    is_not_null = "is_not_null"
    belongs_to = "belongs_to"


# ---------------------------------------------------------------------------
# Semantic references (user-facing terms, not yet grounded)
# ---------------------------------------------------------------------------


class SemanticReference(BaseModel):
    """A reference to a column, value, or computed expression in the user prompt."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal[
        "column_reference",
        "literal_value",
        "range_value",
        "list_value",
        "expression",
        "all_columns",
    ]
    user_term: str = ""
    value: Any = None
    minimum: Any = None
    maximum: Any = None
    values: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Filter predicate structure
# ---------------------------------------------------------------------------


class FilterPredicate(BaseModel):
    """A single filter condition expressed semantically."""

    model_config = ConfigDict(extra="ignore")

    left: SemanticReference
    operator: RelationOperator
    right: SemanticReference | None = None


# ---------------------------------------------------------------------------
# Semantic task and goal models
# ---------------------------------------------------------------------------


class SemanticOperation(BaseModel):
    """The typed operation within a semantic task."""

    model_config = ConfigDict(extra="ignore")

    type: SemanticOperationType


class SemanticTask(BaseModel):
    """One discrete user-requested operation in the semantic plan."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    operation: SemanticOperation
    inputs: list[SemanticReference] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    confidence: float | None = None


class SemanticGoal(BaseModel):
    """A high-level user goal inferred from the prompt."""

    model_config = ConfigDict(extra="ignore")

    description: str
    priority: int = 1


class OutputRequirement(BaseModel):
    """What the user expects as output."""

    model_config = ConfigDict(extra="ignore")

    format: str | None = None
    description: str = ""
    columns: list[SemanticReference] = Field(default_factory=list)


class SemanticConstraint(BaseModel):
    """A constraint the user placed on the operation (e.g. ordering, limits)."""

    model_config = ConfigDict(extra="ignore")

    constraint_type: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class SemanticAmbiguity(BaseModel):
    """An ambiguous requirement that needs clarification."""

    model_config = ConfigDict(extra="ignore")

    description: str
    possible_interpretations: list[str] = Field(default_factory=list)
    source_text: str = ""


class UnsupportedRequirement(BaseModel):
    """A requirement the system cannot fulfill."""

    model_config = ConfigDict(extra="ignore")

    description: str
    reason: str = ""
    source_text: str = ""


# ---------------------------------------------------------------------------
# Top-level semantic intent container
# ---------------------------------------------------------------------------


class SemanticIntent(BaseModel):
    """The complete semantic representation of a user's prompt.

    This is the output of the constrained LLM extraction step, before
    grounding and compilation.
    """

    model_config = ConfigDict(extra="ignore")

    goals: list[SemanticGoal] = Field(default_factory=list)
    tasks: list[SemanticTask] = Field(default_factory=list)
    outputs: list[OutputRequirement] = Field(default_factory=list)
    constraints: list[SemanticConstraint] = Field(default_factory=list)
    ambiguities: list[SemanticAmbiguity] = Field(default_factory=list)
    unsupported_requirements: list[UnsupportedRequirement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Grounding result models
# ---------------------------------------------------------------------------


class ColumnGroundingResult(BaseModel):
    """Result of grounding a user-facing term to an actual column."""

    model_config = ConfigDict(extra="ignore")

    user_term: str
    resolved_column: str | None = None
    confidence: float = 0.0
    resolution_type: Literal[
        "exact_match",
        "case_insensitive_match",
        "normalized_match",
        "semantic_column_match",
        "fuzzy_match",
        "no_match",
    ] = "no_match"
    candidates: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class GroundedSemanticIntent(BaseModel):
    """Semantic intent with all column references resolved."""

    model_config = ConfigDict(extra="ignore")

    intent: SemanticIntent
    grounding_results: list[ColumnGroundingResult] = Field(default_factory=list)
    all_resolved: bool = False
    unresolved_references: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Coverage verification models
# ---------------------------------------------------------------------------


class MissingRequirement(BaseModel):
    """A user requirement that is not represented in the semantic intent."""

    model_config = ConfigDict(extra="ignore")

    description: str
    source_text: str = ""
    suggested_operation: SemanticOperationType | None = None


class ConflictingRequirement(BaseModel):
    """Two requirements that contradict each other."""

    model_config = ConfigDict(extra="ignore")

    description: str
    task_a: str = ""
    task_b: str = ""


class CoverageResult(BaseModel):
    """Result of verifying that the semantic intent covers all user requirements."""

    model_config = ConfigDict(extra="ignore")

    covered: bool
    missing_requirements: list[MissingRequirement] = Field(default_factory=list)
    conflicting_requirements: list[ConflictingRequirement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction metadata (for observability)
# ---------------------------------------------------------------------------


class ExtractionMetadata(BaseModel):
    """Metadata about a semantic extraction run, for debugging and audit."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.0"
    extraction_model: str = ""
    extraction_prompt_version: str = "1.0"
    deterministic_evidence: list[str] = Field(default_factory=list)
    raw_semantic_output: dict[str, Any] = Field(default_factory=dict)
    validated_semantic_output: dict[str, Any] = Field(default_factory=dict)
    coverage_result: CoverageResult | None = None
    repair_result: dict[str, Any] | None = None
    grounding_candidates: list[ColumnGroundingResult] = Field(default_factory=list)
    final_intent_hash: str = ""
