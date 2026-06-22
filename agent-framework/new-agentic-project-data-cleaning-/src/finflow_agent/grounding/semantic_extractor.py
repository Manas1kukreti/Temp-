"""Semantic Extractor for FinFlow's multi-stage pipeline.

Converts raw user prompts into SemanticIntentDraft objects using an LLM.
Preserves ambiguity, classifies generic references, maintains boolean scope,
and ensures every extracted element has typed ProvenanceRef.

The extractor NEVER produces a CanonicalIntent — only SemanticIntentDraft.

Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 8.1, 15.1
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    LLMProviderError,
    LLMValidationError,
    SemanticResolver,
)
from finflow_agent.models.draft import (
    AmbiguityMarker,
    DraftAction,
    DropAction,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    ReferenceKind,
    RenameAction,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
    UnresolvedPredicate,
)
from finflow_agent.models.provenance import (
    PromptSpanProvenance,
    ProvenanceRef,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Words that should be classified as generic_reference (Req 1.3)
GENERIC_WORDS: frozenset[str] = frozenset({
    "field", "fields",
    "column", "columns",
    "attribute", "attributes",
    "entry", "entries",
    "data",
})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when semantic extraction fails.

    This covers both LLM infrastructure failures and response parsing errors.
    Maps to resolution_status=interpretation_failed (never needs_clarification).

    Requirements: 18.1
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Schema Context Model
# ---------------------------------------------------------------------------


class SchemaContext(BaseModel):
    """Optional schema information provided to the extractor for context.

    When available, helps the LLM understand available columns and types
    for better extraction quality.
    """

    model_config = ConfigDict(strict=True)

    column_names: list[str] = Field(
        default_factory=list,
        description="Available column names in the dataset",
    )
    column_types: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column name to inferred dtype",
    )
    dataset_description: str | None = Field(
        default=None,
        description="Optional human-readable dataset description",
    )


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a data-operation semantic intent extractor. Parse the user instruction \
into a structured JSON representation that preserves ALL ambiguity and uses \
typed provenance references.

CRITICAL RULES:
1. PRESERVE AMBIGUITY: If multiple operation interpretations are plausible, \
list ALL candidates in the "ambiguities" array. NEVER commit to a single \
interpretation when uncertain.
2. GENERIC REFERENCES: Words like "field", "column", "attribute", "entry", \
"data" MUST be classified as reference_kind="generic_reference" — NEVER as \
literal column names.
3. BOOLEAN SCOPE: Preserve the connective the user actually wrote.
   - OR lists (e.g., "paypal or cash") MUST stay as a single predicate with \
     operator="in" and a value list.
   - AND lists (e.g., "A and B") MUST stay as separate predicates inside the \
     same logical group with operator="and".
   NEVER collapse AND into a value list or split OR into separate clauses.
4. PROVENANCE: For EVERY extracted element (action, reference, operator, value, \
logical_group), include start_offset and end_offset (Unicode code-point offsets \
in the original prompt) plus the source_text.
5. OUTPUT: Return ONLY a SemanticIntentDraft JSON. Never produce a CanonicalIntent.

REFERENCE_KIND CLASSIFICATION:
- "explicit_name": user used the exact column name (e.g., "payment_method")
- "semantic_concept": user described the column semantically (e.g., "how they paid")
- "generic_reference": user used a generic word (field, column, attribute, entry, data)
- "value_implied": user referenced a value that implies a column (e.g., "the paypal ones")
- "column_group": user referenced a group of columns (e.g., "all numeric columns")

OUTPUT JSON SCHEMA:
{
  "actions": [
    {
      "type": "filter|project|drop|sort|rename",
      ... action-specific fields ...
      "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
    }
  ],
  "ambiguities": [
    {
      "element_path": "actions[0]",
      "candidates": ["filter", "project"],
      "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
    }
  ],
  "ignored_spans": []
}

For filter actions:
{
  "type": "filter",
  "logical_groups": [{
    "operator": "and"|"or",
    "predicates": [{
      "field_ref": {
        "reference_text": "...",
        "reference_kind": "explicit_name|semantic_concept|generic_reference|value_implied|column_group",
        "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
      },
      "operator": "eq|ne|gt|gte|lt|lte|in|not_in|contains|is_null|is_not_null",
      "value": "..." or ["...","..."],
      "negated": false,
      "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
    }],
    "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
  }],
  "provenance": [{"type":"prompt_span","start_offset":N,"end_offset":M,"source_text":"..."}]
}

BOOLEAN SCOPE EXAMPLE:
"filter where payment is paypal or cash" →
  operator="in", value=["paypal","cash"] (SINGLE predicate, NOT two separate predicates)
"""


# ---------------------------------------------------------------------------
# SemanticExtractor
# ---------------------------------------------------------------------------


class SemanticExtractor:
    """Converts raw user prompts into SemanticIntentDraft using an LLM.

    Guarantees:
    - Output is ALWAYS SemanticIntentDraft (never CanonicalIntent)
    - Every extracted element has at least one ProvenanceRef
    - Generic words classified as generic_reference (Req 1.3)
    - Multiple interpretations preserved as ambiguity markers (Req 1.4)
    - Boolean scope preserved — value sets not split (Req 1.5)

    Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 8.1, 15.1
    """

    def __init__(self, resolver: SemanticResolver) -> None:
        """Initialize with an LLM adapter.

        Args:
            resolver: SemanticResolver protocol implementation for LLM calls.
        """
        self._resolver = resolver

    async def extract(
        self,
        prompt: str,
        schema_context: SchemaContext | None = None,
    ) -> SemanticIntentDraft:
        """Extract semantic intent from raw prompt.

        Guarantees:
        - Output is SemanticIntentDraft (never CanonicalIntent)
        - Every extracted element has at least one ProvenanceRef
        - Generic words classified as generic_reference
        - Multiple interpretations preserved as ambiguity markers
        - Boolean scope preserved (value sets not split)

        Args:
            prompt: Raw user prompt text.
            schema_context: Optional schema information for extraction context.

        Returns:
            SemanticIntentDraft with extracted actions and provenance.

        Raises:
            ExtractionError: If extraction fails (LLM or parsing).
        """
        if not prompt or not prompt.strip():
            raise ExtractionError("Empty prompt cannot be extracted")

        messages = self._build_messages(prompt, schema_context)

        try:
            response = await self._resolver.call(
                messages,
                call_site=LLMCallSite.EXTRACTION,
                constraint=DEFAULT_CONSTRAINTS[LLMCallSite.EXTRACTION],
            )
        except (LLMProviderError, LLMValidationError) as exc:
            raise ExtractionError(
                f"LLM extraction failed: {exc}", cause=exc
            ) from exc

        try:
            raw_data = self._parse_response(response.content)
        except (ValueError, KeyError) as exc:
            raise ExtractionError(
                f"Failed to parse LLM extraction response: {exc}", cause=exc
            ) from exc

        draft = self._build_draft(prompt, raw_data)
        return draft

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, prompt: str, schema_context: SchemaContext | None
    ) -> list[dict[str, str]]:
        """Construct the LLM message array for extraction."""
        user_content_parts = [f"USER PROMPT:\n{prompt}"]

        if schema_context:
            if schema_context.column_names:
                col_lines = []
                for col in schema_context.column_names:
                    dtype = schema_context.column_types.get(col, "unknown")
                    col_lines.append(f"  - {col} ({dtype})")
                user_content_parts.append(
                    "AVAILABLE COLUMNS:\n" + "\n".join(col_lines)
                )
            if schema_context.dataset_description:
                user_content_parts.append(
                    f"DATASET DESCRIPTION: {schema_context.dataset_description}"
                )

        user_content_parts.append(
            "Return ONLY the JSON structure. No explanations."
        )

        return [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response content into a dictionary.

        Handles markdown code fences and common formatting issues.
        """
        text = content.strip()
        # Strip code fences
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to extract balanced JSON fragment
            fragment = self._extract_json_fragment(text)
            if fragment:
                data = json.loads(fragment)
            else:
                raise ValueError(
                    f"Could not parse JSON from LLM response: {exc}"
                ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object, got {type(data).__name__}"
            )
        return data

    @staticmethod
    def _extract_json_fragment(text: str) -> str | None:
        """Extract the first balanced JSON object from text."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def _build_draft(
        self, prompt: str, raw_data: dict[str, Any]
    ) -> SemanticIntentDraft:
        """Build a SemanticIntentDraft from parsed LLM output.

        Applies post-processing to ensure:
        - Generic words are classified correctly (Req 1.3)
        - Every element has provenance (Req 1.6, 15.1)
        - Boolean scope is preserved (Req 1.5)
        """
        actions = self._parse_actions(prompt, raw_data.get("actions", []))
        ambiguities = self._parse_ambiguities(
            prompt, raw_data.get("ambiguities", [])
        )
        ignored_spans = self._parse_ignored_spans(
            raw_data.get("ignored_spans", [])
        )

        # Build whole-prompt provenance for the extraction itself
        extraction_provenance: list[ProvenanceRef] = [
            PromptSpanProvenance(
                start_offset=0,
                end_offset=len(prompt),
                source_text=prompt,
            )
        ]

        return SemanticIntentDraft(
            raw_prompt=prompt,
            actions=actions,
            ambiguities=ambiguities,
            ignored_spans=ignored_spans,
            resolution_status=ResolutionStatus.PENDING,
            resolution_origin=ResolutionOrigin.DIRECT,
            extraction_provenance=extraction_provenance,
        )

    def _parse_actions(
        self, prompt: str, raw_actions: list[Any]
    ) -> list[DraftAction]:
        """Parse raw action dicts into typed DraftAction objects."""
        actions: list[DraftAction] = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue
            action_type = raw_action.get("type", "")
            try:
                action = self._parse_single_action(prompt, raw_action, action_type)
                if action is not None:
                    actions.append(action)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed action of type=%r: %s",
                    action_type, exc,
                )
        return actions

    def _parse_single_action(
        self, prompt: str, raw: dict[str, Any], action_type: str
    ) -> DraftAction | None:
        """Parse a single action dict into the appropriate typed action."""
        provenance = self._extract_provenance_list(prompt, raw.get("provenance"))

        if action_type == "filter":
            return self._parse_filter_action(prompt, raw, provenance)
        elif action_type == "project":
            return self._parse_project_action(prompt, raw, provenance)
        elif action_type == "drop":
            return self._parse_drop_action(prompt, raw, provenance)
        elif action_type == "sort":
            return self._parse_sort_action(prompt, raw, provenance)
        elif action_type == "rename":
            return self._parse_rename_action(prompt, raw, provenance)
        else:
            logger.warning("Unknown action type: %r", action_type)
            return None

    def _parse_filter_action(
        self,
        prompt: str,
        raw: dict[str, Any],
        provenance: list[ProvenanceRef],
    ) -> FilterAction:
        """Parse a filter action with logical groups preserving boolean scope."""
        raw_groups = raw.get("logical_groups", [])
        logical_groups: list[LogicalGroup] = []

        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue
            group_prov = self._extract_provenance_list(
                prompt, raw_group.get("provenance")
            )
            predicates = self._parse_predicates(
                prompt, raw_group.get("predicates", [])
            )
            if predicates:
                logical_groups.append(
                    LogicalGroup(
                        operator=raw_group.get("operator", "and"),
                        predicates=predicates,
                        provenance=group_prov,
                    )
                )

        if not logical_groups:
            # Fallback: wrap raw predicates in a single AND group
            raw_preds = raw.get("predicates", [])
            predicates = self._parse_predicates(prompt, raw_preds)
            if predicates:
                logical_groups.append(
                    LogicalGroup(
                        operator="and",
                        predicates=predicates,
                        provenance=provenance,
                    )
                )

        if not logical_groups:
            # Create a minimal valid filter with a placeholder
            logical_groups.append(
                LogicalGroup(
                    operator="and",
                    predicates=[
                        UnresolvedPredicate(
                            field_ref=SemanticColumnReference(
                                reference_text="unknown",
                                reference_kind=ReferenceKind.GENERIC_REFERENCE,
                                provenance=provenance,
                            ),
                            operator="eq",
                            value=None,
                            provenance=provenance,
                        )
                    ],
                    provenance=provenance,
                )
            )

        return FilterAction(logical_groups=logical_groups, provenance=provenance)

    def _parse_predicates(
        self, prompt: str, raw_predicates: list[Any]
    ) -> list[UnresolvedPredicate]:
        """Parse raw predicate dicts into UnresolvedPredicate objects.

        Preserves boolean scope: value lists stay as single predicate (Req 1.5).
        """
        predicates: list[UnresolvedPredicate] = []
        for raw_pred in raw_predicates:
            if not isinstance(raw_pred, dict):
                continue
            pred_prov = self._extract_provenance_list(
                prompt, raw_pred.get("provenance")
            )
            field_ref = self._parse_column_reference(
                prompt, raw_pred.get("field_ref", {})
            )
            # Preserve value as-is (may be a list for "in" operators → Req 1.5)
            value = raw_pred.get("value")
            predicates.append(
                UnresolvedPredicate(
                    field_ref=field_ref,
                    operator=raw_pred.get("operator", "eq"),
                    value=value,
                    negated=bool(raw_pred.get("negated", False)),
                    provenance=pred_prov,
                )
            )
        return predicates

    def _parse_project_action(
        self,
        prompt: str,
        raw: dict[str, Any],
        provenance: list[ProvenanceRef],
    ) -> ProjectAction:
        """Parse a project (select columns) action."""
        columns = self._parse_column_references(
            prompt, raw.get("columns", [])
        )
        if not columns:
            columns = [
                SemanticColumnReference(
                    reference_text="all",
                    reference_kind=ReferenceKind.GENERIC_REFERENCE,
                    provenance=provenance,
                )
            ]
        return ProjectAction(columns=columns, provenance=provenance)

    def _parse_drop_action(
        self,
        prompt: str,
        raw: dict[str, Any],
        provenance: list[ProvenanceRef],
    ) -> DropAction:
        """Parse a drop columns action."""
        columns = self._parse_column_references(
            prompt, raw.get("columns", [])
        )
        if not columns:
            columns = [
                SemanticColumnReference(
                    reference_text="unknown",
                    reference_kind=ReferenceKind.GENERIC_REFERENCE,
                    provenance=provenance,
                )
            ]
        return DropAction(columns=columns, provenance=provenance)

    def _parse_sort_action(
        self,
        prompt: str,
        raw: dict[str, Any],
        provenance: list[ProvenanceRef],
    ) -> SortAction:
        """Parse a sort action."""
        keys = self._parse_column_references(prompt, raw.get("keys", []))
        directions = raw.get("directions", [])
        if not keys:
            keys = [
                SemanticColumnReference(
                    reference_text="unknown",
                    reference_kind=ReferenceKind.GENERIC_REFERENCE,
                    provenance=provenance,
                )
            ]
        # Ensure directions matches keys length
        if len(directions) < len(keys):
            directions = directions + ["asc"] * (len(keys) - len(directions))
        elif len(directions) > len(keys):
            directions = directions[: len(keys)]
        # Validate direction values
        directions = [
            d if d in ("asc", "desc") else "asc" for d in directions
        ]
        return SortAction(keys=keys, directions=directions, provenance=provenance)

    def _parse_rename_action(
        self,
        prompt: str,
        raw: dict[str, Any],
        provenance: list[ProvenanceRef],
    ) -> RenameAction:
        """Parse a rename columns action."""
        raw_mappings = raw.get("mappings", [])
        mappings: list[tuple[SemanticColumnReference, str]] = []
        for item in raw_mappings:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                ref_data, new_name = item
                if isinstance(ref_data, dict):
                    ref = self._parse_column_reference(prompt, ref_data)
                else:
                    ref = self._make_column_ref_from_text(
                        prompt, str(ref_data), provenance
                    )
                mappings.append((ref, str(new_name)))
            elif isinstance(item, dict):
                source = item.get("source", item.get("from", {}))
                target = item.get("target", item.get("to", "unknown"))
                if isinstance(source, dict):
                    ref = self._parse_column_reference(prompt, source)
                else:
                    ref = self._make_column_ref_from_text(
                        prompt, str(source), provenance
                    )
                mappings.append((ref, str(target)))
        if not mappings:
            mappings = [
                (
                    SemanticColumnReference(
                        reference_text="unknown",
                        reference_kind=ReferenceKind.GENERIC_REFERENCE,
                        provenance=provenance,
                    ),
                    "unknown",
                )
            ]
        return RenameAction(mappings=mappings, provenance=provenance)

    def _parse_column_references(
        self, prompt: str, raw_refs: list[Any]
    ) -> list[SemanticColumnReference]:
        """Parse a list of column reference dicts."""
        refs: list[SemanticColumnReference] = []
        for raw_ref in raw_refs:
            if isinstance(raw_ref, dict):
                refs.append(self._parse_column_reference(prompt, raw_ref))
            elif isinstance(raw_ref, str):
                refs.append(
                    self._make_column_ref_from_text(prompt, raw_ref, None)
                )
        return refs

    def _parse_column_reference(
        self, prompt: str, raw_ref: dict[str, Any]
    ) -> SemanticColumnReference:
        """Parse a single column reference dict.

        Applies generic word classification (Req 1.3).
        """
        reference_text = str(raw_ref.get("reference_text", "unknown"))
        raw_kind = raw_ref.get("reference_kind", "")
        provenance = self._extract_provenance_list(
            prompt, raw_ref.get("provenance")
        )

        # Apply generic word classification (Req 1.3)
        reference_kind = self._classify_reference_kind(
            reference_text, raw_kind
        )

        return SemanticColumnReference(
            reference_text=reference_text,
            reference_kind=reference_kind,
            resolved_column=raw_ref.get("resolved_column"),
            confidence=raw_ref.get("confidence"),
            provenance=provenance,
        )

    def _make_column_ref_from_text(
        self,
        prompt: str,
        text: str,
        fallback_provenance: list[ProvenanceRef] | None,
    ) -> SemanticColumnReference:
        """Create a column reference from raw text, finding its span."""
        provenance = self._find_span_provenance(prompt, text)
        if not provenance and fallback_provenance:
            provenance = fallback_provenance
        elif not provenance:
            provenance = [self._whole_prompt_provenance(prompt)]

        kind = self._classify_reference_kind(text, "")
        return SemanticColumnReference(
            reference_text=text,
            reference_kind=kind,
            provenance=provenance,
        )

    def _classify_reference_kind(
        self, reference_text: str, raw_kind: str
    ) -> ReferenceKind:
        """Classify reference kind, ensuring generic words are generic_reference.

        Requirement 1.3: Words like "field", "column", "attribute", "entry",
        "data" MUST be classified as generic_reference.
        """
        # Check if the reference text itself is a generic word
        normalized = reference_text.strip().lower()
        if normalized in GENERIC_WORDS:
            return ReferenceKind.GENERIC_REFERENCE

        # Check if ANY word in the reference text is exclusively a generic word
        words = set(re.findall(r"\b\w+\b", normalized))
        if words and words.issubset(GENERIC_WORDS):
            return ReferenceKind.GENERIC_REFERENCE

        # Trust the LLM's classification if it's valid
        try:
            return ReferenceKind(raw_kind)
        except ValueError:
            # Fallback heuristic
            if normalized in GENERIC_WORDS:
                return ReferenceKind.GENERIC_REFERENCE
            return ReferenceKind.SEMANTIC_CONCEPT

    def _parse_ambiguities(
        self, prompt: str, raw_ambiguities: list[Any]
    ) -> list[AmbiguityMarker]:
        """Parse ambiguity markers from LLM output (Req 1.4)."""
        markers: list[AmbiguityMarker] = []
        for raw_amb in raw_ambiguities:
            if not isinstance(raw_amb, dict):
                continue
            element_path = str(raw_amb.get("element_path", "unknown"))
            candidates = raw_amb.get("candidates", [])
            if not isinstance(candidates, list) or len(candidates) < 1:
                continue
            candidates = [str(c) for c in candidates if c]
            if not candidates:
                continue
            provenance = self._extract_provenance_list(
                prompt, raw_amb.get("provenance")
            )
            markers.append(
                AmbiguityMarker(
                    element_path=element_path,
                    candidates=candidates,
                    provenance=provenance,
                )
            )
        return markers

    def _parse_ignored_spans(
        self, raw_spans: list[Any]
    ) -> list[PromptSpanProvenance]:
        """Parse ignored span records."""
        spans: list[PromptSpanProvenance] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, dict):
                continue
            try:
                spans.append(
                    PromptSpanProvenance(
                        start_offset=int(raw_span.get("start_offset", 0)),
                        end_offset=int(raw_span.get("end_offset", 1)),
                        source_text=str(
                            raw_span.get("source_text", " ")
                        ),
                    )
                )
            except (ValueError, TypeError):
                continue
        return spans

    # ------------------------------------------------------------------
    # Provenance helpers
    # ------------------------------------------------------------------

    def _extract_provenance_list(
        self, prompt: str, raw_provenance: Any
    ) -> list[ProvenanceRef]:
        """Extract provenance list from raw data, guaranteeing at least one entry.

        Requirement 1.6, 15.1: Every element MUST have at least one ProvenanceRef.
        """
        if not raw_provenance or not isinstance(raw_provenance, list):
            return [self._whole_prompt_provenance(prompt)]

        provenance_refs: list[ProvenanceRef] = []
        for raw_prov in raw_provenance:
            if not isinstance(raw_prov, dict):
                continue
            prov_type = raw_prov.get("type", "prompt_span")
            if prov_type == "prompt_span":
                try:
                    start = int(raw_prov.get("start_offset", 0))
                    end = int(raw_prov.get("end_offset", len(prompt)))
                    source_text = str(
                        raw_prov.get("source_text", "")
                    )
                    # Validate and clamp offsets
                    start = max(0, min(start, len(prompt) - 1))
                    end = max(start + 1, min(end, len(prompt)))
                    if not source_text:
                        source_text = prompt[start:end]
                    provenance_refs.append(
                        PromptSpanProvenance(
                            start_offset=start,
                            end_offset=end,
                            source_text=source_text,
                        )
                    )
                except (ValueError, TypeError):
                    continue

        # Guarantee at least one provenance entry (Req 1.6)
        if not provenance_refs:
            provenance_refs = [self._whole_prompt_provenance(prompt)]

        return provenance_refs

    def _find_span_provenance(
        self, prompt: str, text: str
    ) -> list[ProvenanceRef]:
        """Find the span of text within the prompt and create provenance."""
        if not text:
            return []
        lower_prompt = prompt.lower()
        lower_text = text.lower()
        idx = lower_prompt.find(lower_text)
        if idx >= 0:
            end = idx + len(text)
            return [
                PromptSpanProvenance(
                    start_offset=idx,
                    end_offset=end,
                    source_text=prompt[idx:end],
                )
            ]
        return []

    @staticmethod
    def _whole_prompt_provenance(prompt: str) -> PromptSpanProvenance:
        """Create a provenance entry spanning the entire prompt."""
        return PromptSpanProvenance(
            start_offset=0,
            end_offset=max(1, len(prompt)),
            source_text=prompt[:100] if len(prompt) > 100 else prompt,
        )
