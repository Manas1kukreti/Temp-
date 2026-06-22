"""Shared grounding package for FinFlow's semantic pipeline.

This package contains the deterministic-first grounding components responsible
for resolving column references and filter predicates to physical dataset columns.

Key components:
- candidate_generator: Shared Candidate Generation Layer (token overlap, value-concept,
  semantic-type scoring)
- scoring: Deterministic scoring utilities
- evidence: Evidence models (positive/negative) and grounding configuration
- column_grounder: Column Grounder for standalone references (projections, sorts, drops, renames)
- predicate_grounder: Predicate Grounder for filter predicates (field + operator + value)
- schema_service: Schema Service with layered structural/value-evidence cache
- preflight_loader: Preflight Data Loader (read-only file profiling)
- llm_adapter: SemanticResolver LLM protocol and retry policies
- verification: Post-LLM deterministic verification and tie-breaking
"""

from finflow_agent.grounding.candidate_generator import (
    CandidateGenerator,
    ScoringWeights,
)
from finflow_agent.grounding.column_grounder import (
    ColumnGrounder,
    ReferenceContextError,
)
from finflow_agent.grounding.evidence import (
    ColumnGroundingResult,
    GroundingConfig,
    GroundingMethod,
    PredicateGroundingResult,
    ScoredCandidate,
)
from finflow_agent.grounding.predicate_grounder import PredicateGrounder
from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    LLMConstraint,
    LLMProviderError,
    LLMResponse,
    LLMValidationError,
    RetryPolicy,
    SemanticResolver,
)
from finflow_agent.grounding.preflight_loader import (
    FileTooLargeError,
    PreflightConfig,
    PreflightDataLoader,
    UnsupportedFormatError,
)
from finflow_agent.grounding.schema_service import (
    ColumnRole,
    ColumnSemanticType,
    SchemaInferenceResult,
    SchemaService,
)
from finflow_agent.grounding.scoring import (
    compute_name_similarity,
    compute_semantic_type_alignment,
    compute_token_overlap,
    compute_value_concept_match,
    tokenize,
)
from finflow_agent.grounding.semantic_extractor import (
    ExtractionError,
    SchemaContext,
    SemanticExtractor,
)
from finflow_agent.grounding.verification import (
    PostLLMVerification,
    TieBreakingPolicy,
    apply_tie_breaking_policy,
    verify_llm_selection,
)

__all__ = [
    "CandidateGenerator",
    "ColumnGrounder",
    "ColumnGroundingResult",
    "ColumnRole",
    "ColumnSemanticType",
    "DEFAULT_CONSTRAINTS",
    "ExtractionError",
    "FileTooLargeError",
    "GroundingConfig",
    "GroundingMethod",
    "LLMCallSite",
    "LLMConstraint",
    "LLMProviderError",
    "LLMResponse",
    "LLMValidationError",
    "PostLLMVerification",
    "PredicateGroundingResult",
    "PredicateGrounder",
    "PreflightConfig",
    "PreflightDataLoader",
    "ReferenceContextError",
    "RetryPolicy",
    "SchemaContext",
    "SchemaInferenceResult",
    "SchemaService",
    "ScoredCandidate",
    "ScoringWeights",
    "SemanticExtractor",
    "SemanticResolver",
    "TieBreakingPolicy",
    "UnsupportedFormatError",
    "apply_tie_breaking_policy",
    "compute_name_similarity",
    "compute_semantic_type_alignment",
    "compute_token_overlap",
    "compute_value_concept_match",
    "tokenize",
    "verify_llm_selection",
]
