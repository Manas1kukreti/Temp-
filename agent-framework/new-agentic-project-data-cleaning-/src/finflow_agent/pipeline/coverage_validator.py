"""Deterministic Coverage Validator for FinFlow's semantic pipeline.

Performs structural validation of SemanticIntentDraft without any keyword
analysis or semantic re-interpretation. Shadow LLM mode runs behind a feature
flag with no authority over the deterministic result.

Key responsibilities:
- Action-to-reference completeness checks
- Provenance completeness enforcement
- Material span coverage verification
- Duplicate/contradictory action detection
- Boolean-group preservation validation
- Unresolved-reference declaration validity
- Shadow LLM comparison (non-authoritative, behind feature flag)

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.draft import (
    DraftAction,
    DropAction,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    RenameAction,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
)
from finflow_agent.models.envelope import ShadowComparisonMetric
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.pipeline.feature_flags import FeatureFlags
from finflow_agent.pipeline.observability import ShadowModeRecorder


# ---------------------------------------------------------------------------
# Failure models
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """Category of structural failure detected by coverage validation."""

    ACTION_REFERENCE_INCOMPLETE = "action_reference_incomplete"
    PROVENANCE_MISSING = "provenance_missing"
    SPAN_NOT_LINKED = "span_not_linked"
    DUPLICATE_ACTION = "duplicate_action"
    CONTRADICTORY_ACTION = "contradictory_action"
    BOOLEAN_GROUP_INVALID = "boolean_group_invalid"
    UNRESOLVED_REFERENCE_INVALID = "unresolved_reference_invalid"


class StructuralFailure(BaseModel):
    """Describes a specific structural gap found during coverage validation.

    Each failure identifies the category, location (element path within the draft),
    and a human-readable description of the issue.

    Requirements: 3.1, 3.6
    """

    model_config = ConfigDict(strict=True)

    category: FailureCategory = Field(
        ..., description="Category of the structural failure"
    )
    element_path: str = Field(
        ..., description="JSON path to the failing element in the draft"
    )
    description: str = Field(
        ..., description="Human-readable description of the structural gap"
    )


class CoverageValidationResult(BaseModel):
    """Result of deterministic structural coverage validation.

    Contains a pass/fail indicator and the list of structural failures found.
    An empty failures list corresponds to passed=True.

    Requirements: 3.1
    """

    model_config = ConfigDict(strict=True)

    passed: bool = Field(
        ..., description="True if no structural failures were detected"
    )
    failures: list[StructuralFailure] = Field(
        default_factory=list,
        description="List of structural failures (empty when passed=True)",
    )


# ---------------------------------------------------------------------------
# CoverageValidator implementation
# ---------------------------------------------------------------------------


class CoverageValidator:
    """Deterministic structural coverage validator.

    Performs only structural/deterministic checks. Does NOT perform keyword
    analysis or semantic re-interpretation (Req 3.3).

    Shadow LLM mode is gated by FeatureFlags.ENABLE_LLM_COVERAGE_SHADOW
    and has no authority over the deterministic result (Req 3.4).

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
    """

    def __init__(
        self,
        feature_flags: FeatureFlags,
        shadow_recorder: ShadowModeRecorder | None = None,
    ) -> None:
        self._feature_flags = feature_flags
        self._shadow_recorder = shadow_recorder or ShadowModeRecorder()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def validate(
        self, draft: SemanticIntentDraft, prompt: str
    ) -> CoverageValidationResult:
        """Check structural correctness of the extracted draft.

        Performs deterministic checks only — no keyword analysis or semantic
        re-interpretation (Req 3.3).

        Returns CoverageValidationResult with pass/fail and failure details.
        On failure, downstream pipeline routes to Semantic_Repair (Req 3.6).
        """
        failures: list[StructuralFailure] = []

        failures.extend(self._check_action_reference_completeness(draft))
        failures.extend(self._check_provenance_completeness(draft))
        failures.extend(self._check_material_span_coverage(draft, prompt))
        failures.extend(self._check_duplicate_actions(draft))
        failures.extend(self._check_contradictory_actions(draft))
        failures.extend(self._check_boolean_group_preservation(draft))
        failures.extend(self._check_unresolved_references(draft))

        passed = len(failures) == 0
        return CoverageValidationResult(passed=passed, failures=failures)

    async def shadow_llm_coverage(
        self, draft: SemanticIntentDraft, prompt: str
    ) -> ShadowComparisonMetric | None:
        """Run LLM coverage in shadow mode (no authority).

        Returns None if shadow mode is disabled via feature flag.
        When enabled, runs the deterministic validation and records a
        ShadowComparisonMetric comparing deterministic vs. LLM results.

        The LLM result has NO authority — it is recorded for observability
        only (Req 3.4). Emits ShadowComparisonMetric when active (Req 3.5).
        """
        if not self._feature_flags.ENABLE_LLM_COVERAGE_SHADOW:
            return None

        # Run deterministic validation to get the authoritative result
        deterministic_result = self.validate(draft, prompt)

        # Shadow LLM: placeholder for actual LLM invocation
        # In production this would call an LLM to assess coverage
        # For now we record that the LLM was unavailable
        llm_result: bool | None = None
        llm_gaps: list[str] = []

        deterministic_gaps = [f.description for f in deterministic_result.failures]

        # Emit the ShadowComparisonMetric (Req 3.5)
        metric = self._shadow_recorder.record_comparison(
            deterministic_result=deterministic_result.passed,
            llm_result=llm_result,
            deterministic_gaps=deterministic_gaps,
            llm_gaps=llm_gaps,
        )

        return metric

    # -------------------------------------------------------------------
    # Structural checks (Req 3.1 — deterministic, no keyword analysis)
    # -------------------------------------------------------------------

    def _check_action_reference_completeness(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Every action must have at least one reference."""
        failures: list[StructuralFailure] = []

        for idx, action in enumerate(draft.actions):
            path = f"actions[{idx}]"
            refs = self._extract_references_from_action(action)

            if not refs:
                failures.append(
                    StructuralFailure(
                        category=FailureCategory.ACTION_REFERENCE_INCOMPLETE,
                        element_path=path,
                        description=(
                            f"Action at {path} (type={action.type}) "
                            f"has no column references"
                        ),
                    )
                )

        return failures

    def _check_provenance_completeness(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Every semantic element must have at least one ProvenanceRef."""
        failures: list[StructuralFailure] = []

        for idx, action in enumerate(draft.actions):
            action_path = f"actions[{idx}]"

            # Check action-level provenance
            if not action.provenance:
                failures.append(
                    StructuralFailure(
                        category=FailureCategory.PROVENANCE_MISSING,
                        element_path=action_path,
                        description=(
                            f"Action at {action_path} has no provenance references"
                        ),
                    )
                )

            # Check column reference provenance within actions
            refs = self._extract_references_from_action(action)
            for ref_idx, ref in enumerate(refs):
                ref_path = f"{action_path}.references[{ref_idx}]"
                if not ref.provenance:
                    failures.append(
                        StructuralFailure(
                            category=FailureCategory.PROVENANCE_MISSING,
                            element_path=ref_path,
                            description=(
                                f"Reference '{ref.reference_text}' at {ref_path} "
                                f"has no provenance"
                            ),
                        )
                    )

            # Check logical group provenance for filter actions
            if isinstance(action, FilterAction):
                for grp_idx, group in enumerate(action.logical_groups):
                    grp_path = f"{action_path}.logical_groups[{grp_idx}]"
                    if not group.provenance:
                        failures.append(
                            StructuralFailure(
                                category=FailureCategory.PROVENANCE_MISSING,
                                element_path=grp_path,
                                description=(
                                    f"Logical group at {grp_path} has no provenance"
                                ),
                            )
                        )

                    for pred_idx, pred in enumerate(group.predicates):
                        pred_path = f"{grp_path}.predicates[{pred_idx}]"
                        if not pred.provenance:
                            failures.append(
                                StructuralFailure(
                                    category=FailureCategory.PROVENANCE_MISSING,
                                    element_path=pred_path,
                                    description=(
                                        f"Predicate at {pred_path} has no provenance"
                                    ),
                                )
                            )

        # Check ambiguity marker provenance
        for amb_idx, amb in enumerate(draft.ambiguities):
            amb_path = f"ambiguities[{amb_idx}]"
            if not amb.provenance:
                failures.append(
                    StructuralFailure(
                        category=FailureCategory.PROVENANCE_MISSING,
                        element_path=amb_path,
                        description=(
                            f"Ambiguity marker at {amb_path} has no provenance"
                        ),
                    )
                )

        return failures

    def _check_material_span_coverage(
        self, draft: SemanticIntentDraft, prompt: str
    ) -> list[StructuralFailure]:
        """Every material source span from extraction must be linked to an element.

        A span is "linked" if it is covered by at least one ProvenanceRef
        (from an action, reference, predicate, or ambiguity) or explicitly
        listed in ignored_spans. (Req 3.2)
        """
        failures: list[StructuralFailure] = []

        if not prompt:
            return failures

        # Collect all covered character ranges from provenance refs
        covered_ranges: list[tuple[int, int]] = []
        covered_ranges.extend(self._collect_provenance_spans(draft))

        # Collect ranges from ignored_spans
        for span in draft.ignored_spans:
            covered_ranges.append((span.start_offset, span.end_offset))

        # Identify non-whitespace character positions that must be covered
        material_positions: set[int] = set()
        for i, ch in enumerate(prompt):
            if not ch.isspace():
                material_positions.add(i)

        # Determine which positions are covered
        covered_positions: set[int] = set()
        for start, end in covered_ranges:
            for pos in range(start, end):
                covered_positions.add(pos)

        # Find uncovered material positions
        uncovered = material_positions - covered_positions
        if uncovered:
            # Group into contiguous spans for clearer reporting
            sorted_uncovered = sorted(uncovered)
            spans: list[tuple[int, int]] = []
            span_start = sorted_uncovered[0]
            span_end = sorted_uncovered[0] + 1

            for pos in sorted_uncovered[1:]:
                if pos == span_end:
                    span_end = pos + 1
                else:
                    spans.append((span_start, span_end))
                    span_start = pos
                    span_end = pos + 1
            spans.append((span_start, span_end))

            for start, end in spans:
                uncovered_text = prompt[start:end]
                failures.append(
                    StructuralFailure(
                        category=FailureCategory.SPAN_NOT_LINKED,
                        element_path=f"prompt[{start}:{end}]",
                        description=(
                            f"Material span '{uncovered_text}' "
                            f"(offset {start}-{end}) not linked to any element"
                        ),
                    )
                )

        return failures

    def _check_duplicate_actions(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Detect duplicate actions: same action type targeting the same columns."""
        failures: list[StructuralFailure] = []

        # Track (action_type, frozenset_of_column_texts) for duplicate detection
        seen: dict[tuple[str, frozenset[str]], int] = {}

        for idx, action in enumerate(draft.actions):
            refs = self._extract_references_from_action(action)
            col_key = frozenset(r.reference_text for r in refs)
            action_key = (action.type, col_key)

            if action_key in seen:
                original_idx = seen[action_key]
                failures.append(
                    StructuralFailure(
                        category=FailureCategory.DUPLICATE_ACTION,
                        element_path=f"actions[{idx}]",
                        description=(
                            f"Duplicate action: actions[{idx}] (type={action.type}) "
                            f"duplicates actions[{original_idx}] on same columns"
                        ),
                    )
                )
            else:
                seen[action_key] = idx

        return failures

    def _check_contradictory_actions(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Detect contradictory actions (e.g., project and drop same column)."""
        failures: list[StructuralFailure] = []

        # Contradictory pairs: actions that conflict when targeting same columns
        contradictory_pairs = [("project", "drop")]

        # Build map: action_type → list of (index, column_texts)
        action_columns: dict[str, list[tuple[int, set[str]]]] = {}
        for idx, action in enumerate(draft.actions):
            refs = self._extract_references_from_action(action)
            col_texts = {r.reference_text for r in refs}
            if action.type not in action_columns:
                action_columns[action.type] = []
            action_columns[action.type].append((idx, col_texts))

        # Check each contradictory pair
        for type_a, type_b in contradictory_pairs:
            if type_a not in action_columns or type_b not in action_columns:
                continue

            for idx_a, cols_a in action_columns[type_a]:
                for idx_b, cols_b in action_columns[type_b]:
                    overlap = cols_a & cols_b
                    if overlap:
                        failures.append(
                            StructuralFailure(
                                category=FailureCategory.CONTRADICTORY_ACTION,
                                element_path=f"actions[{idx_a}],actions[{idx_b}]",
                                description=(
                                    f"Contradictory actions: {type_a} (actions[{idx_a}]) "
                                    f"and {type_b} (actions[{idx_b}]) target "
                                    f"overlapping columns: {sorted(overlap)}"
                                ),
                            )
                        )

        return failures

    def _check_boolean_group_preservation(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Boolean/logical groups must have at least one predicate."""
        failures: list[StructuralFailure] = []

        for idx, action in enumerate(draft.actions):
            if not isinstance(action, FilterAction):
                continue

            for grp_idx, group in enumerate(action.logical_groups):
                grp_path = f"actions[{idx}].logical_groups[{grp_idx}]"

                # Each logical group must have predicates (this is also
                # enforced by Pydantic min_length=1, but we validate
                # structurally in case of manual construction)
                if not group.predicates:
                    failures.append(
                        StructuralFailure(
                            category=FailureCategory.BOOLEAN_GROUP_INVALID,
                            element_path=grp_path,
                            description=(
                                f"Logical group at {grp_path} has no predicates"
                            ),
                        )
                    )

                # Each predicate must have a valid field_ref
                for pred_idx, pred in enumerate(group.predicates):
                    pred_path = f"{grp_path}.predicates[{pred_idx}]"
                    if not pred.field_ref.reference_text:
                        failures.append(
                            StructuralFailure(
                                category=FailureCategory.BOOLEAN_GROUP_INVALID,
                                element_path=pred_path,
                                description=(
                                    f"Predicate at {pred_path} has empty "
                                    f"field reference text"
                                ),
                            )
                        )

        return failures

    def _check_unresolved_references(
        self, draft: SemanticIntentDraft
    ) -> list[StructuralFailure]:
        """Unresolved reference declarations must be valid.

        A reference is "unresolved" if resolved_column is None. These must
        still have a valid reference_text and reference_kind.
        """
        failures: list[StructuralFailure] = []

        for idx, action in enumerate(draft.actions):
            refs = self._extract_references_from_action(action)
            for ref_idx, ref in enumerate(refs):
                ref_path = f"actions[{idx}].references[{ref_idx}]"

                # Unresolved references must still have valid structure
                if ref.resolved_column is None:
                    if not ref.reference_text:
                        failures.append(
                            StructuralFailure(
                                category=FailureCategory.UNRESOLVED_REFERENCE_INVALID,
                                element_path=ref_path,
                                description=(
                                    f"Unresolved reference at {ref_path} "
                                    f"has empty reference_text"
                                ),
                            )
                        )

        return failures

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _extract_references_from_action(
        self, action: DraftAction
    ) -> list[SemanticColumnReference]:
        """Extract all SemanticColumnReferences from an action."""
        if isinstance(action, ProjectAction):
            return list(action.columns)
        elif isinstance(action, DropAction):
            return list(action.columns)
        elif isinstance(action, SortAction):
            return list(action.keys)
        elif isinstance(action, RenameAction):
            return [mapping[0] for mapping in action.mappings]
        elif isinstance(action, FilterAction):
            refs: list[SemanticColumnReference] = []
            for group in action.logical_groups:
                for predicate in group.predicates:
                    refs.append(predicate.field_ref)
            return refs
        return []

    def _collect_provenance_spans(
        self, draft: SemanticIntentDraft
    ) -> list[tuple[int, int]]:
        """Collect all PromptSpanProvenance offset ranges from the draft."""
        spans: list[tuple[int, int]] = []

        def _extract_spans_from_provenance(
            provenance_list: list[Any],
        ) -> None:
            for prov in provenance_list:
                if hasattr(prov, "type") and prov.type == "prompt_span":
                    spans.append((prov.start_offset, prov.end_offset))

        # Extraction-level provenance
        _extract_spans_from_provenance(draft.extraction_provenance)

        # Action-level provenance and nested references
        for action in draft.actions:
            _extract_spans_from_provenance(action.provenance)

            if isinstance(action, FilterAction):
                for group in action.logical_groups:
                    _extract_spans_from_provenance(group.provenance)
                    for pred in group.predicates:
                        _extract_spans_from_provenance(pred.provenance)
                        _extract_spans_from_provenance(pred.field_ref.provenance)
            elif isinstance(action, ProjectAction):
                for col in action.columns:
                    _extract_spans_from_provenance(col.provenance)
            elif isinstance(action, DropAction):
                for col in action.columns:
                    _extract_spans_from_provenance(col.provenance)
            elif isinstance(action, SortAction):
                for key in action.keys:
                    _extract_spans_from_provenance(key.provenance)
            elif isinstance(action, RenameAction):
                for mapping in action.mappings:
                    _extract_spans_from_provenance(mapping[0].provenance)

        # Ambiguity marker provenance
        for amb in draft.ambiguities:
            _extract_spans_from_provenance(amb.provenance)

        return spans
