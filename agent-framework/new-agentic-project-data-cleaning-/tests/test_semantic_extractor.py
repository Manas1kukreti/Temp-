"""Unit tests for the SemanticExtractor.

Tests cover:
- Produces SemanticIntentDraft (not CanonicalIntent)
- Generic word classification (Req 1.3)
- Ambiguity preservation (Req 1.4)
- Boolean scope preservation (Req 1.5)
- Provenance on every element (Req 1.6, 15.1)
- ExtractionError on failures
- SchemaContext integration

Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 8.1, 15.1
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from finflow_agent.grounding.llm_adapter import (
    LLMCallSite,
    LLMProviderError,
    LLMResponse,
    LLMValidationError,
)
from finflow_agent.grounding.semantic_extractor import (
    GENERIC_WORDS,
    ExtractionError,
    SchemaContext,
    SemanticExtractor,
)
from finflow_agent.models.draft import (
    ReferenceKind,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticIntentDraft,
)
from finflow_agent.models.provenance import PromptSpanProvenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolver(response_content: str) -> Any:
    """Create a mock SemanticResolver that returns the given content."""
    resolver = AsyncMock()
    resolver.call.return_value = LLMResponse(
        content=response_content,
        parsed=None,
        call_site=LLMCallSite.EXTRACTION,
        latency_ms=50.0,
    )
    return resolver


def _filter_response(prompt: str = "filter where payment is paypal or cash") -> str:
    """Build a typical filter extraction response preserving boolean scope."""
    return json.dumps({
        "actions": [
            {
                "type": "filter",
                "logical_groups": [
                    {
                        "operator": "and",
                        "predicates": [
                            {
                                "field_ref": {
                                    "reference_text": "payment",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 13,
                                            "end_offset": 20,
                                            "source_text": "payment",
                                        }
                                    ],
                                },
                                "operator": "in",
                                "value": ["paypal", "cash"],
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 13,
                                        "end_offset": 41,
                                        "source_text": "payment is paypal or cash",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 41,
                                "source_text": prompt,
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 41,
                        "source_text": prompt,
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _ambiguous_response() -> str:
    """Build a response with ambiguity markers (Req 1.4)."""
    return json.dumps({
        "actions": [
            {
                "type": "project",
                "columns": [
                    {
                        "reference_text": "amount",
                        "reference_kind": "explicit_name",
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 5,
                                "end_offset": 11,
                                "source_text": "amount",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 20,
                        "source_text": "show amount by date",
                    }
                ],
            }
        ],
        "ambiguities": [
            {
                "element_path": "actions[0]",
                "candidates": ["project", "sort"],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 20,
                        "source_text": "show amount by date",
                    }
                ],
            }
        ],
        "ignored_spans": [],
    })


def _generic_ref_response() -> str:
    """Build a response where the LLM returns a generic word reference."""
    return json.dumps({
        "actions": [
            {
                "type": "drop",
                "columns": [
                    {
                        "reference_text": "column",
                        "reference_kind": "explicit_name",  # LLM misclassifies
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 5,
                                "end_offset": 11,
                                "source_text": "column",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 20,
                        "source_text": "drop column with nulls",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSemanticExtractorBasics:
    """Test basic extraction behavior."""

    @pytest.mark.asyncio
    async def test_produces_semantic_intent_draft(self):
        """Extraction always produces SemanticIntentDraft, never CanonicalIntent."""
        prompt = "filter where payment is paypal or cash"
        resolver = _make_resolver(_filter_response(prompt))
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        assert isinstance(result, SemanticIntentDraft)
        assert result.resolution_status == ResolutionStatus.PENDING
        assert result.resolution_origin == ResolutionOrigin.DIRECT
        assert result.raw_prompt == prompt

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_extraction_error(self):
        """Empty prompts raise ExtractionError immediately."""
        resolver = _make_resolver("{}")
        extractor = SemanticExtractor(resolver)

        with pytest.raises(ExtractionError, match="Empty prompt"):
            await extractor.extract("")

        with pytest.raises(ExtractionError, match="Empty prompt"):
            await extractor.extract("   ")

    @pytest.mark.asyncio
    async def test_llm_provider_error_raises_extraction_error(self):
        """LLM provider failures produce ExtractionError."""
        resolver = AsyncMock()
        resolver.call.side_effect = LLMProviderError(
            "timeout", error_type="timeout", call_site="extraction"
        )
        extractor = SemanticExtractor(resolver)

        with pytest.raises(ExtractionError, match="LLM extraction failed"):
            await extractor.extract("filter rows")

    @pytest.mark.asyncio
    async def test_llm_validation_error_raises_extraction_error(self):
        """LLM validation failures produce ExtractionError."""
        resolver = AsyncMock()
        resolver.call.side_effect = LLMValidationError(
            "bad json", call_site="extraction"
        )
        extractor = SemanticExtractor(resolver)

        with pytest.raises(ExtractionError, match="LLM extraction failed"):
            await extractor.extract("sort by date")

    @pytest.mark.asyncio
    async def test_invalid_json_raises_extraction_error(self):
        """Non-JSON LLM response produces ExtractionError."""
        resolver = _make_resolver("this is not json at all")
        extractor = SemanticExtractor(resolver)

        with pytest.raises(ExtractionError, match="Failed to parse"):
            await extractor.extract("show me the data")


class TestGenericWordClassification:
    """Req 1.3: Generic words classified as generic_reference."""

    @pytest.mark.asyncio
    async def test_generic_word_overrides_llm_classification(self):
        """Even if LLM says explicit_name, generic words → generic_reference."""
        prompt = "drop column with nulls"
        resolver = _make_resolver(_generic_ref_response())
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        # The "column" reference must be generic_reference
        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.type == "drop"
        col_ref = action.columns[0]  # type: ignore[attr-defined]
        assert col_ref.reference_kind == ReferenceKind.GENERIC_REFERENCE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("word", list(GENERIC_WORDS))
    async def test_all_generic_words_classified(self, word: str):
        """All defined generic words are classified correctly."""
        response = json.dumps({
            "actions": [
                {
                    "type": "project",
                    "columns": [
                        {
                            "reference_text": word,
                            "reference_kind": "explicit_name",
                            "provenance": [
                                {
                                    "type": "prompt_span",
                                    "start_offset": 5,
                                    "end_offset": 5 + len(word),
                                    "source_text": word,
                                }
                            ],
                        }
                    ],
                    "provenance": [
                        {
                            "type": "prompt_span",
                            "start_offset": 0,
                            "end_offset": 10 + len(word),
                            "source_text": f"show {word}",
                        }
                    ],
                }
            ],
            "ambiguities": [],
            "ignored_spans": [],
        })
        resolver = _make_resolver(response)
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(f"show {word} values")

        col = result.actions[0].columns[0]  # type: ignore[attr-defined]
        assert col.reference_kind == ReferenceKind.GENERIC_REFERENCE


class TestAmbiguityPreservation:
    """Req 1.4: Multiple interpretations preserved as ambiguity markers."""

    @pytest.mark.asyncio
    async def test_ambiguity_markers_preserved(self):
        """Ambiguity markers from LLM output are preserved in draft."""
        prompt = "show amount by date"
        resolver = _make_resolver(_ambiguous_response())
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        assert len(result.ambiguities) == 1
        amb = result.ambiguities[0]
        assert amb.element_path == "actions[0]"
        assert "project" in amb.candidates
        assert "sort" in amb.candidates
        assert len(amb.provenance) >= 1


class TestBooleanScopePreservation:
    """Req 1.5: Value sets preserved as single predicate, not split."""

    @pytest.mark.asyncio
    async def test_value_list_preserved_as_single_predicate(self):
        """'paypal or cash' stays as one predicate with value list."""
        prompt = "filter where payment is paypal or cash"
        resolver = _make_resolver(_filter_response(prompt))
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.type == "filter"
        # Single logical group with single predicate
        groups = action.logical_groups  # type: ignore[attr-defined]
        assert len(groups) == 1
        preds = groups[0].predicates
        assert len(preds) == 1
        # Value is a list (not split into separate clauses)
        assert preds[0].operator == "in"
        assert preds[0].value == ["paypal", "cash"]


class TestProvenanceCompleteness:
    """Req 1.6, 15.1: Every element has at least one ProvenanceRef."""

    @pytest.mark.asyncio
    async def test_all_actions_have_provenance(self):
        """Every action has at least one provenance entry."""
        prompt = "filter where payment is paypal or cash"
        resolver = _make_resolver(_filter_response(prompt))
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        for action in result.actions:
            assert len(action.provenance) >= 1
            for prov in action.provenance:
                assert prov.type == "prompt_span"

    @pytest.mark.asyncio
    async def test_column_references_have_provenance(self):
        """Every column reference has at least one provenance entry."""
        prompt = "show amount by date"
        resolver = _make_resolver(_ambiguous_response())
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        action = result.actions[0]
        for col in action.columns:  # type: ignore[attr-defined]
            assert len(col.provenance) >= 1

    @pytest.mark.asyncio
    async def test_extraction_provenance_covers_whole_prompt(self):
        """Extraction provenance spans the full prompt."""
        prompt = "sort by date ascending"
        response = json.dumps({
            "actions": [
                {
                    "type": "sort",
                    "keys": [
                        {
                            "reference_text": "date",
                            "reference_kind": "explicit_name",
                            "provenance": [
                                {
                                    "type": "prompt_span",
                                    "start_offset": 8,
                                    "end_offset": 12,
                                    "source_text": "date",
                                }
                            ],
                        }
                    ],
                    "directions": ["asc"],
                    "provenance": [
                        {
                            "type": "prompt_span",
                            "start_offset": 0,
                            "end_offset": 22,
                            "source_text": prompt,
                        }
                    ],
                }
            ],
            "ambiguities": [],
            "ignored_spans": [],
        })
        resolver = _make_resolver(response)
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        # Extraction provenance should cover the whole prompt
        assert len(result.extraction_provenance) >= 1
        ext_prov = result.extraction_provenance[0]
        assert ext_prov.type == "prompt_span"
        assert ext_prov.start_offset == 0  # type: ignore[union-attr]
        assert ext_prov.end_offset == len(prompt)  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_fallback_provenance_when_llm_omits_it(self):
        """When LLM omits provenance, extractor adds whole-prompt fallback."""
        prompt = "drop empty columns"
        response = json.dumps({
            "actions": [
                {
                    "type": "drop",
                    "columns": [
                        {
                            "reference_text": "empty columns",
                            "reference_kind": "column_group",
                            # No provenance from LLM
                        }
                    ],
                    # No provenance from LLM
                }
            ],
            "ambiguities": [],
        })
        resolver = _make_resolver(response)
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        action = result.actions[0]
        # Action should have fallback provenance
        assert len(action.provenance) >= 1
        # Column ref should have fallback provenance
        col = action.columns[0]  # type: ignore[attr-defined]
        assert len(col.provenance) >= 1


class TestSchemaContext:
    """Test SchemaContext integration."""

    @pytest.mark.asyncio
    async def test_schema_context_passed_to_llm(self):
        """SchemaContext info is included in LLM messages."""
        prompt = "filter by amount > 100"
        resolver = _make_resolver(json.dumps({
            "actions": [],
            "ambiguities": [],
            "ignored_spans": [],
        }))
        extractor = SemanticExtractor(resolver)
        ctx = SchemaContext(
            column_names=["amount", "date", "category"],
            column_types={"amount": "float64", "date": "datetime", "category": "object"},
            dataset_description="Financial transactions dataset",
        )

        await extractor.extract(prompt, schema_context=ctx)

        # Verify the resolver was called with messages containing schema info
        call_args = resolver.call.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "amount" in user_msg
        assert "float64" in user_msg
        assert "Financial transactions dataset" in user_msg

    @pytest.mark.asyncio
    async def test_extraction_without_schema_context(self):
        """Extraction works without schema context."""
        prompt = "show all rows"
        resolver = _make_resolver(json.dumps({
            "actions": [
                {
                    "type": "project",
                    "columns": [
                        {
                            "reference_text": "all",
                            "reference_kind": "column_group",
                            "provenance": [
                                {
                                    "type": "prompt_span",
                                    "start_offset": 5,
                                    "end_offset": 8,
                                    "source_text": "all",
                                }
                            ],
                        }
                    ],
                    "provenance": [
                        {
                            "type": "prompt_span",
                            "start_offset": 0,
                            "end_offset": 13,
                            "source_text": prompt,
                        }
                    ],
                }
            ],
            "ambiguities": [],
            "ignored_spans": [],
        }))
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt, schema_context=None)

        assert isinstance(result, SemanticIntentDraft)
        assert len(result.actions) == 1


class TestCodeFenceHandling:
    """Test that markdown code fences in LLM response are handled."""

    @pytest.mark.asyncio
    async def test_strips_code_fences(self):
        """LLM response wrapped in code fences is parsed correctly."""
        prompt = "show amount"
        raw = json.dumps({
            "actions": [
                {
                    "type": "project",
                    "columns": [
                        {
                            "reference_text": "amount",
                            "reference_kind": "explicit_name",
                            "provenance": [
                                {
                                    "type": "prompt_span",
                                    "start_offset": 5,
                                    "end_offset": 11,
                                    "source_text": "amount",
                                }
                            ],
                        }
                    ],
                    "provenance": [
                        {
                            "type": "prompt_span",
                            "start_offset": 0,
                            "end_offset": 11,
                            "source_text": prompt,
                        }
                    ],
                }
            ],
            "ambiguities": [],
            "ignored_spans": [],
        })
        wrapped = f"```json\n{raw}\n```"
        resolver = _make_resolver(wrapped)
        extractor = SemanticExtractor(resolver)

        result = await extractor.extract(prompt)

        assert isinstance(result, SemanticIntentDraft)
        assert len(result.actions) == 1
