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

Additionally, this module provides `run_semantic_pipeline_with_clarification()`
which integrates with the ClarificationService to route resolvable ambiguities
to interactive clarification sessions rather than quarantine.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

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

    @property
    def awaiting_clarification(self) -> bool:
        return self.resolution_status == "awaiting_clarification"


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


# ---------------------------------------------------------------------------
# Clarification-integrated pipeline
# ---------------------------------------------------------------------------


def _compute_ambiguity_score(result: SemanticExtractionResult) -> float:
    """Compute an ambiguity score from the pipeline result.

    The ambiguity score indicates how resolvable the ambiguity is:
    - Higher scores (>= threshold) mean the ambiguity has good candidates
      and is suitable for interactive clarification.
    - Lower scores (< threshold) mean the ambiguity is too severe for
      clarification and should go directly to quarantine.

    Score computation:
    - If grounded_intent exists with unresolved references that have candidates,
      the score is the average of the best candidate confidence across unresolved refs.
    - If no grounding info is available (e.g., extraction/coverage failure),
      the score defaults to 0.0 (quarantine directly).
    """
    grounded = result.grounded_intent
    if grounded is None:
        # No grounding results — ambiguity is not resolvable via clarification
        return 0.0

    # Gather confidence scores from grounding results that need clarification
    candidate_confidences: list[float] = []
    for gr in grounded.grounding_results:
        if gr.needs_clarification or gr.resolved_column is None:
            # Use the confidence score (which represents best candidate match)
            if gr.confidence > 0.0:
                candidate_confidences.append(gr.confidence)
            elif gr.candidates:
                # Has candidates but no confidence scored — moderately resolvable
                candidate_confidences.append(0.5)
            else:
                # No candidates at all — not resolvable
                candidate_confidences.append(0.0)

    if not candidate_confidences:
        # No unresolved references found in grounding (shouldn't happen if
        # resolution_status is needs_clarification, but handle gracefully)
        return 0.0

    return sum(candidate_confidences) / len(candidate_confidences)


async def run_semantic_pipeline_with_clarification(
    raw_prompt: str,
    source_columns: list[str],
    *,
    submission_id: UUID,
    db: Any,
    column_types: dict[str, str] | None = None,
    output_format: str = "xlsx",
    llm_call: Any = None,
    use_llm_coverage: bool = False,
    intent: Any = None,
    intent_package: Any = None,
) -> SemanticExtractionResult:
    """Run the semantic pipeline with clarification service integration.

    This function wraps `run_semantic_pipeline` and adds interactive ambiguity
    resolution routing. After the pipeline produces a "needs_clarification"
    result, it:

    1. Computes an ambiguity score from the grounding results.
    2. Calls ClarificationService.initiate_session() with the score.
    3. If a session is created (score >= threshold), halts further pipeline
       processing and returns early with resolution_status="awaiting_clarification".
    4. If no session is created (score < threshold), returns the original
       "needs_clarification" result for existing quarantine logic to handle.

    Parameters
    ----------
    raw_prompt : str
        The raw user instruction.
    source_columns : list[str]
        Available columns in the dataset.
    submission_id : UUID
        The UUID of the submission being processed.
    db : AsyncSession
        SQLAlchemy async database session for ClarificationService.
    column_types : dict | None
        Optional column type mapping.
    output_format : str
        Desired output format.
    llm_call : callable | None
        Optional custom LLM callable for testing.
    use_llm_coverage : bool
        Whether to use LLM for coverage check.
    intent : Any | None
        Optional pre-existing CanonicalIntent (used if available).
    intent_package : Any | None
        Optional IntentPackage for schema resolution context.

    Returns
    -------
    SemanticExtractionResult
        The extraction result. If clarification was initiated, resolution_status
        will be "awaiting_clarification" and pipeline processing is halted.

    Requirements
    -----------
    1.1: Score >= threshold → session created + status "awaiting_clarification"
    1.2: Score < threshold → quarantined without session (handled by caller)
    """
    # --- Run the base semantic pipeline ---
    result = await run_semantic_pipeline(
        raw_prompt,
        source_columns,
        column_types=column_types,
        output_format=output_format,
        llm_call=llm_call,
        use_llm_coverage=use_llm_coverage,
    )

    # --- If pipeline resolved successfully, return as-is ---
    if result.resolution_status != "needs_clarification":
        return result

    # --- Compute ambiguity score from grounding results ---
    ambiguity_score = _compute_ambiguity_score(result)

    # --- Attempt to initiate a clarification session ---
    try:
        from app.services.clarification_service import ClarificationService

        service = ClarificationService(db)

        # Use the provided intent/intent_package, or fall back to pipeline results
        clarification_intent = intent or (
            result.grounded_intent.intent.model_dump(mode="json")
            if result.grounded_intent and result.grounded_intent.intent
            else result.semantic_intent.model_dump(mode="json")
            if result.semantic_intent
            else {}
        )
        clarification_intent_package = intent_package or result.grounded_intent

        session = await service.initiate_session(
            submission_id=submission_id,
            intent=clarification_intent,
            intent_package=clarification_intent_package,
            ambiguity_score=ambiguity_score,
        )

        if session is not None:
            # Session created — halt further pipeline processing and return early
            # The submission status has been transitioned to "awaiting_clarification"
            # by ClarificationService.initiate_session()
            logger.info(
                "Clarification session %s created for submission %s (ambiguity_score=%.3f)",
                session.id,
                submission_id,
                ambiguity_score,
            )
            return SemanticExtractionResult(
                resolution_status="awaiting_clarification",
                semantic_intent=result.semantic_intent,
                grounded_intent=result.grounded_intent,
                coverage_result=result.coverage_result,
                repair_notes=result.repair_notes,
                metadata=result.metadata,
                evidence=[
                    f"Clarification session initiated (score={ambiguity_score:.3f}). "
                    f"Awaiting user input for {len(result.grounded_intent.unresolved_references) if result.grounded_intent else 0} "
                    f"unresolved reference(s)."
                ],
            )

        # Session NOT created — ambiguity_score below threshold.
        # Return original result with needs_clarification for quarantine logic.
        logger.info(
            "Ambiguity score %.3f below threshold for submission %s; proceeding to quarantine.",
            ambiguity_score,
            submission_id,
        )
        return result

    except Exception as e:
        # Graceful degradation: if clarification service is unavailable,
        # fall back to existing quarantine behavior (design spec requirement).
        logger.warning(
            "ClarificationService unavailable for submission %s: %s. "
            "Falling back to quarantine.",
            submission_id,
            e,
        )
        return result
