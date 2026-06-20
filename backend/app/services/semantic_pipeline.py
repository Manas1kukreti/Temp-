"""Hybrid semantic extraction pipeline.

This module orchestrates the full hybrid extraction flow:

    raw prompt
    → deterministic fast-path attempt
    → if complex: constrained LLM semantic extraction
    → normalize
    → ground against schema
    → coverage check
    → repair if needed (max 1)
    → if complete: compile to canonical actions
    → if incomplete: return needs_clarification

The existing canonical intent system remains the final output format.
This module REPLACES the primary understanding mechanism (regex) with
semantic LLM extraction while keeping deterministic logic as fast-path
and validation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.services.semantic_models import (
    CoverageResult,
    ExtractionMetadata,
    GroundedSemanticIntent,
    SemanticIntent,
)
from app.services.semantic_extractor import (
    SEMANTIC_EXTRACTOR_VERSION,
    SEMANTIC_SCHEMA_VERSION,
    extract_semantic_intent,
    extract_semantic_intent_sync,
)
from app.services.semantic_normalizer import (
    SEMANTIC_NORMALIZER_VERSION,
    normalize_semantic_intent,
)
from app.services.semantic_grounding import (
    SEMANTIC_GROUNDING_VERSION,
    ground_semantic_intent,
)
from app.services.semantic_coverage import (
    COVERAGE_CHECKER_VERSION,
    check_coverage_deterministic,
    check_coverage_with_llm,
)
from app.services.semantic_repair import (
    SEMANTIC_REPAIR_VERSION,
    repair_semantic_intent,
)
from app.services.semantic_compiler import (
    SEMANTIC_COMPILER_VERSION,
    SemanticCompilationError,
    compile_semantic_to_canonical,
)

logger = logging.getLogger(__name__)

# Version string for observability
HYBRID_PIPELINE_VERSION = "1.0"


class SemanticExtractionResult:
    """Result of the hybrid semantic extraction pipeline."""

    def __init__(
        self,
        *,
        canonical_actions: list[dict[str, Any]] | None = None,
        output_format: str = "xlsx",
        resolution_status: str = "resolved",
        evidence: list[str] | None = None,
        assumptions: list[str] | None = None,
        repair_notes: list[str] | None = None,
        semantic_intent: SemanticIntent | None = None,
        grounded_intent: GroundedSemanticIntent | None = None,
        coverage_result: CoverageResult | None = None,
        metadata: ExtractionMetadata | None = None,
        error: str | None = None,
    ):
        self.canonical_actions = canonical_actions or []
        self.output_format = output_format
        self.resolution_status = resolution_status
        self.evidence = evidence or []
        self.assumptions = assumptions or []
        self.repair_notes = repair_notes or []
        self.semantic_intent = semantic_intent
        self.grounded_intent = grounded_intent
        self.coverage_result = coverage_result
        self.metadata = metadata
        self.error = error

    @property
    def success(self) -> bool:
        return self.resolution_status in ("resolved", "repaired")

    @property
    def needs_clarification(self) -> bool:
        return self.resolution_status == "needs_clarification"


async def run_semantic_pipeline(
    raw_prompt: str,
    source_columns: list[str],
    *,
    column_types: dict[str, str] | None = None,
    output_format: str = "xlsx",
    llm_call: Any = None,
    use_llm_coverage: bool = False,
) -> SemanticExtractionResult:
    """Run the full hybrid semantic extraction pipeline (async).

    This is the main entry point for semantic extraction.

    Parameters
    ----------
    raw_prompt : str
        The raw user instruction.
    source_columns : list[str]
        Available columns in the dataset.
    column_types : dict | None
        Optional column type mapping.
    output_format : str
        Desired output format.
    llm_call : callable | None
        Optional custom LLM callable for testing.
    use_llm_coverage : bool
        Whether to use LLM for coverage check (slower but more thorough).

    Returns
    -------
    SemanticExtractionResult
        The extraction result with canonical actions or clarification status.
    """
    metadata = ExtractionMetadata(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        extraction_model="groq/llama-3.1-70b",
        extraction_prompt_version=SEMANTIC_EXTRACTOR_VERSION,
    )

    # ---------------------------------------------------------------
    # Step 1: Constrained LLM semantic extraction
    # ---------------------------------------------------------------
    try:
        raw_intent = await extract_semantic_intent(
            raw_prompt,
            source_columns,
            column_types,
            llm_call=llm_call,
        )
    except ValueError as e:
        logger.error("Semantic extraction failed: %s", e)
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            error=str(e),
            metadata=metadata,
        )

    metadata.raw_semantic_output = raw_intent.model_dump(mode="json")

    # ---------------------------------------------------------------
    # Step 2: Normalize semantic intent
    # ---------------------------------------------------------------
    normalized_intent = normalize_semantic_intent(raw_intent)

    # ---------------------------------------------------------------
    # Step 3: Coverage verification (deterministic)
    # ---------------------------------------------------------------
    coverage = check_coverage_deterministic(raw_prompt, normalized_intent)

    # Optional: LLM coverage check for complex prompts
    if not coverage.covered and use_llm_coverage:
        coverage = await check_coverage_with_llm(
            raw_prompt, normalized_intent, llm_call=llm_call
        )

    metadata.coverage_result = coverage

    # ---------------------------------------------------------------
    # Step 4: Bounded repair if needed (max 1 attempt)
    # ---------------------------------------------------------------
    repair_notes: list[str] = []
    if not coverage.covered:
        normalized_intent, repair_notes = await repair_semantic_intent(
            raw_prompt,
            normalized_intent,
            coverage,
            source_columns,
            llm_call=llm_call,
        )

        # Re-check coverage after repair
        coverage = check_coverage_deterministic(raw_prompt, normalized_intent)
        metadata.coverage_result = coverage

        if not coverage.covered:
            # Still incomplete after repair — fail closed
            return SemanticExtractionResult(
                resolution_status="needs_clarification",
                semantic_intent=normalized_intent,
                coverage_result=coverage,
                repair_notes=repair_notes,
                evidence=["Semantic coverage verification failed after repair attempt."],
                metadata=metadata,
                error="Extraction incomplete: " + "; ".join(
                    r.description for r in coverage.missing_requirements
                ),
            )

    metadata.validated_semantic_output = normalized_intent.model_dump(mode="json")

    # ---------------------------------------------------------------
    # Step 5: Ground column references
    # ---------------------------------------------------------------
    grounded = ground_semantic_intent(normalized_intent, source_columns, column_types)
    metadata.grounding_candidates = grounded.grounding_results

    if not grounded.all_resolved:
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            semantic_intent=normalized_intent,
            grounded_intent=grounded,
            coverage_result=coverage,
            repair_notes=repair_notes,
            evidence=[
                f"Unresolved column references: {grounded.unresolved_references}"
            ],
            metadata=metadata,
            error=f"Cannot resolve columns: {grounded.unresolved_references}",
        )

    # ---------------------------------------------------------------
    # Step 6: Compile to canonical actions
    # ---------------------------------------------------------------
    try:
        compiled = compile_semantic_to_canonical(
            grounded, output_format=output_format
        )
    except SemanticCompilationError as e:
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            semantic_intent=normalized_intent,
            grounded_intent=grounded,
            coverage_result=coverage,
            repair_notes=repair_notes,
            evidence=[f"Compilation failed: {e}"],
            metadata=metadata,
            error=str(e),
        )

    resolution_status = "repaired" if repair_notes else "resolved"

    return SemanticExtractionResult(
        canonical_actions=compiled.get("actions", []),
        output_format=output_format,
        resolution_status=resolution_status,
        evidence=compiled.get("evidence", []),
        assumptions=compiled.get("assumptions", []),
        repair_notes=repair_notes,
        semantic_intent=normalized_intent,
        grounded_intent=grounded,
        coverage_result=coverage,
        metadata=metadata,
    )


def run_semantic_pipeline_sync(
    raw_prompt: str,
    source_columns: list[str],
    *,
    column_types: dict[str, str] | None = None,
    output_format: str = "xlsx",
    llm_call: Any = None,
) -> SemanticExtractionResult:
    """Synchronous version of the semantic pipeline.

    Uses deterministic extraction + coverage only (no LLM calls).
    For sync contexts where async LLM calls are not available,
    the caller can provide a synchronous llm_call.
    """
    metadata = ExtractionMetadata(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        extraction_prompt_version=SEMANTIC_EXTRACTOR_VERSION,
    )

    # Step 1: LLM extraction (sync)
    try:
        raw_intent = extract_semantic_intent_sync(
            raw_prompt,
            source_columns,
            column_types,
            llm_call=llm_call,
        )
    except ValueError as e:
        logger.error("Sync semantic extraction failed: %s", e)
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            error=str(e),
            metadata=metadata,
        )

    metadata.raw_semantic_output = raw_intent.model_dump(mode="json")

    # Step 2: Normalize
    normalized_intent = normalize_semantic_intent(raw_intent)

    # Step 3: Coverage (deterministic only in sync mode)
    coverage = check_coverage_deterministic(raw_prompt, normalized_intent)
    metadata.coverage_result = coverage

    if not coverage.covered:
        # In sync mode, we can't do async repair — return needs_clarification
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            semantic_intent=normalized_intent,
            coverage_result=coverage,
            evidence=["Coverage verification failed (sync mode, no repair available)."],
            metadata=metadata,
            error="Extraction incomplete: " + "; ".join(
                r.description for r in coverage.missing_requirements
            ),
        )

    metadata.validated_semantic_output = normalized_intent.model_dump(mode="json")

    # Step 4: Ground
    grounded = ground_semantic_intent(normalized_intent, source_columns, column_types)
    metadata.grounding_candidates = grounded.grounding_results

    if not grounded.all_resolved:
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            semantic_intent=normalized_intent,
            grounded_intent=grounded,
            coverage_result=coverage,
            metadata=metadata,
            error=f"Cannot resolve columns: {grounded.unresolved_references}",
        )

    # Step 5: Compile
    try:
        compiled = compile_semantic_to_canonical(grounded, output_format=output_format)
    except SemanticCompilationError as e:
        return SemanticExtractionResult(
            resolution_status="needs_clarification",
            semantic_intent=normalized_intent,
            grounded_intent=grounded,
            coverage_result=coverage,
            metadata=metadata,
            error=str(e),
        )

    return SemanticExtractionResult(
        canonical_actions=compiled.get("actions", []),
        output_format=output_format,
        resolution_status="resolved",
        evidence=compiled.get("evidence", []),
        semantic_intent=normalized_intent,
        grounded_intent=grounded,
        coverage_result=coverage,
        metadata=metadata,
    )
