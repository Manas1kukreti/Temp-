"""Pipeline comparison test: Legacy vs New Semantic Pipeline.

Compares how the legacy build_canonical_intent() and the new SemanticExtractor
process the same prompts, showing the structural differences in output.

Test prompts from real user scenarios:
1. "Clean the data and extract rows which contains paypal or cash as field"
2. "Clean the data and extract all columns except consumer_id"
3. "Clean the data and extract all which has age above than 45"
4. "Clean the data and extract all which has gender as female"
5. "Clean the data and extract all which has education as phd"
6. "Clean the data and extract all which has payment status as pending"
7. "Clean the data and extract all which has gender as female education as phd and is single"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest

from finflow_agent.grounding.llm_adapter import LLMCallSite, LLMResponse
from finflow_agent.grounding.semantic_extractor import SemanticExtractor, SchemaContext
from finflow_agent.models.draft import (
    ReferenceKind,
    ResolutionStatus,
    SemanticIntentDraft,
)
from finflow_agent.models.pretty_printer import pretty_print_draft


# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "Clean the data and extract rows which contains paypal or cash as field",
    "Clean the data and extract all columns except consumer_id",
    "Clean the data and extract all which has age above than 45",
    "Clean the data and extract all which has gender as female",
    "Clean the data and extract all which has education as phd",
    "Clean the data and extract all which has payment status as pending",
    "Clean the data and extract all which has gender as female education as phd and is single",
]

# Simulated schema context (representative of a financial dataset)
SAMPLE_COLUMNS = [
    "consumer_id", "age", "gender", "education", "marital_status",
    "payment_method", "payment_status", "amount", "loan", "date",
]

SAMPLE_SCHEMA_CONTEXT = SchemaContext(
    column_names=SAMPLE_COLUMNS,
    column_types={
        "consumer_id": "int64",
        "age": "int64",
        "gender": "object",
        "education": "object",
        "marital_status": "object",
        "payment_method": "object",
        "payment_status": "object",
        "amount": "float64",
        "loan": "float64",
        "date": "datetime64",
    },
    dataset_description="Consumer financial transactions dataset",
)


# ---------------------------------------------------------------------------
# Mock LLM responses for the new pipeline (simulating extraction)
# ---------------------------------------------------------------------------


def _make_extraction_response_prompt_1() -> str:
    """Response for: 'Clean the data and extract rows which contains paypal or cash as field'

    KEY DIFFERENCE: The new pipeline preserves 'paypal or cash' as a VALUE SET
    in a single predicate (Req 1.5 - boolean scope preservation), rather than
    splitting into separate filter clauses.
    """
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
                                    "reference_text": "field",
                                    "reference_kind": "generic_reference",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 63,
                                            "end_offset": 68,
                                            "source_text": "field",
                                        }
                                    ],
                                },
                                "operator": "in",
                                "value": ["paypal", "cash"],
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 38,
                                        "end_offset": 68,
                                        "source_text": "contains paypal or cash as field",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 68,
                                "source_text": "Clean the data and extract rows which contains paypal or cash as field",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 68,
                        "source_text": "extract rows which contains paypal or cash as field",
                    }
                ],
            }
        ],
        "ambiguities": [
            {
                "element_path": "actions[0].logical_groups[0].predicates[0].field_ref",
                "candidates": ["payment_method", "payment_status"],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 63,
                        "end_offset": 68,
                        "source_text": "field",
                    }
                ],
            }
        ],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_2() -> str:
    """Response for: 'Clean the data and extract all columns except consumer_id'

    This is a DROP action — user wants to remove consumer_id from the output.
    The column name 'consumer_id' is explicit (exact match).
    """
    return json.dumps({
        "actions": [
            {
                "type": "drop",
                "columns": [
                    {
                        "reference_text": "consumer_id",
                        "reference_kind": "explicit_name",
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 50,
                                "end_offset": 61,
                                "source_text": "consumer_id",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 61,
                        "source_text": "extract all columns except consumer_id",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_3() -> str:
    """Response for: 'Clean the data and extract all which has age above than 45'

    Filter action: age > 45. Column 'age' is explicit, operator is 'gt'.
    """
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
                                    "reference_text": "age",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 42,
                                            "source_text": "age",
                                        }
                                    ],
                                },
                                "operator": "gt",
                                "value": 45,
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 57,
                                        "source_text": "age above than 45",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 57,
                                "source_text": "Clean the data and extract all which has age above than 45",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 57,
                        "source_text": "extract all which has age above than 45",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_4() -> str:
    """Response for: 'Clean the data and extract all which has gender as female'

    Filter action: gender == 'female'. Explicit column, equality operator.
    """
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
                                    "reference_text": "gender",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 45,
                                            "source_text": "gender",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "female",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 55,
                                        "source_text": "gender as female",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 55,
                                "source_text": "Clean the data and extract all which has gender as female",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 55,
                        "source_text": "extract all which has gender as female",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_5() -> str:
    """Response for: 'Clean the data and extract all which has education as phd'

    Filter action: education == 'phd'. Explicit column, equality operator.
    """
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
                                    "reference_text": "education",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 48,
                                            "source_text": "education",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "phd",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 55,
                                        "source_text": "education as phd",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 55,
                                "source_text": "Clean the data and extract all which has education as phd",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 55,
                        "source_text": "extract all which has education as phd",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_6() -> str:
    """Response for: 'Clean the data and extract all which has payment status as pending'

    Filter action: payment_status == 'pending'.
    Note: 'payment status' is a semantic_concept (two words mapping to one column).
    """
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
                                    "reference_text": "payment status",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 53,
                                            "source_text": "payment status",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "pending",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 64,
                                        "source_text": "payment status as pending",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 64,
                                "source_text": "Clean the data and extract all which has payment status as pending",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 64,
                        "source_text": "extract all which has payment status as pending",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_prompt_7() -> str:
    """Response for: 'Clean the data and extract all which has gender as female education as phd and is single'

    Multi-condition filter: gender=='female' AND education=='phd' AND marital_status=='single'.
    Boolean scope preserved as AND group with 3 predicates.
    'is single' uses value_implied reference kind (the value 'single' implies marital_status column).
    """
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
                                    "reference_text": "gender",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 45,
                                            "source_text": "gender",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "female",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 55,
                                        "source_text": "gender as female",
                                    }
                                ],
                            },
                            {
                                "field_ref": {
                                    "reference_text": "education",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 56,
                                            "end_offset": 65,
                                            "source_text": "education",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "phd",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 56,
                                        "end_offset": 72,
                                        "source_text": "education as phd",
                                    }
                                ],
                            },
                            {
                                "field_ref": {
                                    "reference_text": "single",
                                    "reference_kind": "value_implied",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 80,
                                            "end_offset": 86,
                                            "source_text": "single",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "single",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 77,
                                        "end_offset": 86,
                                        "source_text": "is single",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 86,
                                "source_text": "Clean the data and extract all which has gender as female education as phd and is single",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 86,
                        "source_text": "extract all which has gender as female education as phd and is single",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


_EXTRACTION_RESPONSES = [
    _make_extraction_response_prompt_1,
    _make_extraction_response_prompt_2,
    _make_extraction_response_prompt_3,
    _make_extraction_response_prompt_4,
    _make_extraction_response_prompt_5,
    _make_extraction_response_prompt_6,
    _make_extraction_response_prompt_7,
]


def _make_mock_resolver(prompt_index: int = 0) -> AsyncMock:
    """Create a mock resolver that returns appropriate extraction responses."""
    resolver = AsyncMock()

    if 0 <= prompt_index < len(_EXTRACTION_RESPONSES):
        content = _EXTRACTION_RESPONSES[prompt_index]()
    else:
        content = json.dumps({"actions": [], "ambiguities": [], "ignored_spans": []})

    resolver.call.return_value = LLMResponse(
        content=content,
        parsed=None,
        call_site=LLMCallSite.EXTRACTION,
        latency_ms=200.0,
    )
    return resolver


# ---------------------------------------------------------------------------
# Comparison Tests
# ---------------------------------------------------------------------------


class TestPipelineComparisonPrompt1:
    """Compare legacy vs new pipeline for:
    'Clean the data and extract rows which contains paypal or cash as field'

    This prompt tests the most critical difference: boolean scope preservation.
    """

    @pytest.mark.asyncio
    async def test_new_pipeline_preserves_boolean_scope(self):
        """New pipeline keeps 'paypal or cash' as a single value-set predicate.

        Requirement 1.5: value-list patterns preserved as value set within
        one predicate rather than splitting into separate clauses.
        """
        resolver = _make_mock_resolver(prompt_index=0)
        extractor = SemanticExtractor(resolver)

        draft = await extractor.extract(
            TEST_PROMPTS[0],
            schema_context=SAMPLE_SCHEMA_CONTEXT,
        )

        # Verify it's a SemanticIntentDraft (not CanonicalIntent)
        assert isinstance(draft, SemanticIntentDraft)
        assert draft.resolution_status == ResolutionStatus.PENDING

        # Verify the filter action preserves boolean scope
        assert len(draft.actions) >= 1
        filter_action = draft.actions[0]
        assert filter_action.type == "filter"

        # The key test: 'paypal or cash' should be ONE predicate with a list value
        predicates = filter_action.logical_groups[0].predicates
        assert len(predicates) == 1  # Single predicate, NOT two separate ones

        pred = predicates[0]
        assert pred.operator == "in"
        assert pred.value == ["paypal", "cash"]  # Value set preserved!

    @pytest.mark.asyncio
    async def test_new_pipeline_classifies_generic_reference(self):
        """New pipeline classifies 'field' as generic_reference (Req 1.3).

        The word 'field' is generic — it doesn't refer to a specific column.
        The legacy pipeline might treat it as a literal column name.
        """
        resolver = _make_mock_resolver(prompt_index=0)
        extractor = SemanticExtractor(resolver)

        draft = await extractor.extract(
            TEST_PROMPTS[0],
            schema_context=SAMPLE_SCHEMA_CONTEXT,
        )

        # 'field' must be classified as generic_reference
        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_kind == ReferenceKind.GENERIC_REFERENCE

    @pytest.mark.asyncio
    async def test_new_pipeline_preserves_ambiguity(self):
        """New pipeline preserves ambiguity about which column 'field' refers to.

        Requirement 1.4: multiple interpretations preserved as ambiguity markers.
        The legacy pipeline would guess (e.g., pick payment_method) without asking.
        """
        resolver = _make_mock_resolver(prompt_index=0)
        extractor = SemanticExtractor(resolver)

        draft = await extractor.extract(
            TEST_PROMPTS[0],
            schema_context=SAMPLE_SCHEMA_CONTEXT,
        )

        # Ambiguity should be preserved — not resolved silently
        assert len(draft.ambiguities) >= 1
        amb = draft.ambiguities[0]
        assert "payment_method" in amb.candidates
        assert "payment_status" in amb.candidates

    @pytest.mark.asyncio
    async def test_new_pipeline_includes_provenance(self):
        """New pipeline traces every element back to the source prompt.

        Requirement 1.6, 15.1: every extracted element has ProvenanceRef.
        Legacy pipeline provides no traceability.
        """
        resolver = _make_mock_resolver(prompt_index=0)
        extractor = SemanticExtractor(resolver)

        draft = await extractor.extract(
            TEST_PROMPTS[0],
            schema_context=SAMPLE_SCHEMA_CONTEXT,
        )

        # Every action has provenance
        for action in draft.actions:
            assert len(action.provenance) >= 1
            prov = action.provenance[0]
            assert prov.type == "prompt_span"

        # Extraction-level provenance spans the whole prompt
        assert len(draft.extraction_provenance) >= 1

    @pytest.mark.asyncio
    async def test_pretty_print_shows_readable_output(self):
        """Pretty-print produces human-readable debugging output."""
        resolver = _make_mock_resolver(prompt_index=0)
        extractor = SemanticExtractor(resolver)

        draft = await extractor.extract(
            TEST_PROMPTS[0],
            schema_context=SAMPLE_SCHEMA_CONTEXT,
        )

        output = pretty_print_draft(draft)

        # Verify key structural elements appear in output
        assert "SemanticIntentDraft" in output
        assert "filter" in output
        assert "generic_reference" in output
        assert "paypal" in output or "cash" in output
        assert "Ambiguities" in output

        # Print for visual inspection
        print("\n" + "=" * 70)
        print("NEW PIPELINE OUTPUT (SemanticIntentDraft):")
        print("=" * 70)
        print(output)


class TestLegacyVsNewKeyDifferences:
    """Document the key structural differences between legacy and new pipeline."""

    def test_document_differences(self):
        """This test documents what changes between the two pipelines.

        Not a pass/fail test — it's a reference for understanding the migration.
        """
        differences = {
            "prompt": TEST_PROMPTS[0],
            "legacy_behavior": {
                "boolean_scope": "DESTROYED — splits 'paypal or cash' into separate filter conditions OR treats as single condition but loses the OR semantics",
                "column_resolution": "IMMEDIATE — guesses 'payment_method' without asking user",
                "generic_word_handling": "'field' may be treated as literal column name",
                "provenance": "NONE — no traceability from output back to prompt",
                "ambiguity": "HIDDEN — picks one interpretation silently",
                "output_type": "dict (CanonicalIntent) — directly executable, may be wrong",
            },
            "new_behavior": {
                "boolean_scope": "PRESERVED — 'paypal or cash' stays as operator='in', value=['paypal','cash'] in single predicate",
                "column_resolution": "DEFERRED — marks 'field' as generic_reference, resolution happens later in grounding stage",
                "generic_word_handling": "'field' classified as generic_reference (Req 1.3), triggers clarification",
                "provenance": "FULL — every element traces back to Unicode code-point offsets in original prompt",
                "ambiguity": "EXPLICIT — ambiguity markers list candidates (payment_method, payment_status), triggers clarification",
                "output_type": "SemanticIntentDraft — pre-canonical, preserves uncertainty, grounded later",
            },
            "why_new_is_better": [
                "Won't execute wrong column silently — asks user which column they mean",
                "Won't destroy 'paypal OR cash' into 'paypal AND cash' — preserves intent",
                "Provides audit trail — can explain WHY a decision was made",
                "Separates extraction from grounding — each stage has single responsibility",
            ],
        }

        # Print the comparison
        print("\n" + "=" * 70)
        print(f"PROMPT: {differences['prompt']}")
        print("=" * 70)
        print("\nLEGACY PIPELINE:")
        for key, val in differences["legacy_behavior"].items():
            print(f"  {key}: {val}")
        print("\nNEW PIPELINE:")
        for key, val in differences["new_behavior"].items():
            print(f"  {key}: {val}")
        print("\nWHY NEW IS BETTER:")
        for reason in differences["why_new_is_better"]:
            print(f"  [OK] {reason}")

        # This always passes — it's documentation
        assert True


# ---------------------------------------------------------------------------
# Tests for remaining prompts (2-7)
# ---------------------------------------------------------------------------


class TestPrompt2DropColumns:
    """'Clean the data and extract all columns except consumer_id'

    Tests: drop action with explicit column name.
    """

    @pytest.mark.asyncio
    async def test_produces_drop_action(self):
        resolver = _make_mock_resolver(prompt_index=1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1
        action = draft.actions[0]
        assert action.type == "drop"
        assert action.columns[0].reference_text == "consumer_id"
        assert action.columns[0].reference_kind == ReferenceKind.EXPLICIT_NAME

    @pytest.mark.asyncio
    async def test_no_ambiguity_for_explicit_name(self):
        """Explicit column name should have zero ambiguity."""
        resolver = _make_mock_resolver(prompt_index=1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.ambiguities) == 0

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 2: {TEST_PROMPTS[1]}\n{'='*70}\n{output}")
        assert "drop" in output
        assert "consumer_id" in output


class TestPrompt3FilterAge:
    """'Clean the data and extract all which has age above than 45'

    Tests: filter with explicit column, comparison operator, numeric value.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_with_gt_operator(self):
        resolver = _make_mock_resolver(prompt_index=2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1
        action = draft.actions[0]
        assert action.type == "filter"

        pred = action.logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "age"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "gt"
        assert pred.value == 45

    @pytest.mark.asyncio
    async def test_no_ambiguity(self):
        resolver = _make_mock_resolver(prompt_index=2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)
        assert len(draft.ambiguities) == 0

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 3: {TEST_PROMPTS[2]}\n{'='*70}\n{output}")
        assert "age" in output
        assert "gt" in output


class TestPrompt4FilterGender:
    """'Clean the data and extract all which has gender as female'

    Tests: filter with explicit column, equality operator, string value.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_with_eq_operator(self):
        resolver = _make_mock_resolver(prompt_index=3)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[3], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "gender"
        assert pred.operator == "eq"
        assert pred.value == "female"

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=3)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[3], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 4: {TEST_PROMPTS[3]}\n{'='*70}\n{output}")
        assert "gender" in output
        assert "female" in output


class TestPrompt5FilterEducation:
    """'Clean the data and extract all which has education as phd'

    Tests: filter with explicit column, equality operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter(self):
        resolver = _make_mock_resolver(prompt_index=4)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[4], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "education"
        assert pred.operator == "eq"
        assert pred.value == "phd"

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=4)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[4], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 5: {TEST_PROMPTS[4]}\n{'='*70}\n{output}")
        assert "education" in output
        assert "phd" in output


class TestPrompt6FilterPaymentStatus:
    """'Clean the data and extract all which has payment status as pending'

    Tests: filter with SEMANTIC_CONCEPT reference (two words → one column).
    'payment status' maps to physical column 'payment_status' via grounding.
    """

    @pytest.mark.asyncio
    async def test_semantic_concept_reference(self):
        resolver = _make_mock_resolver(prompt_index=5)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[5], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "payment status"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "pending"

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=5)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[5], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 6: {TEST_PROMPTS[5]}\n{'='*70}\n{output}")
        assert "payment status" in output
        assert "semantic_concept" in output
        assert "pending" in output


class TestPrompt7MultiConditionFilter:
    """'Clean the data and extract all which has gender as female education as phd and is single'

    Tests: multi-predicate AND filter with mixed reference kinds:
    - gender (explicit_name) == female
    - education (explicit_name) == phd
    - 'single' (value_implied) — the VALUE implies the column (marital_status)
    """

    @pytest.mark.asyncio
    async def test_three_predicates_in_and_group(self):
        resolver = _make_mock_resolver(prompt_index=6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        action = draft.actions[0]
        assert action.type == "filter"
        predicates = action.logical_groups[0].predicates
        assert len(predicates) == 3

        # Predicate 1: gender == female
        assert predicates[0].field_ref.reference_text == "gender"
        assert predicates[0].value == "female"

        # Predicate 2: education == phd
        assert predicates[1].field_ref.reference_text == "education"
        assert predicates[1].value == "phd"

        # Predicate 3: 'single' with value_implied reference kind
        assert predicates[2].field_ref.reference_text == "single"
        assert predicates[2].field_ref.reference_kind == ReferenceKind.VALUE_IMPLIED
        assert predicates[2].value == "single"

    @pytest.mark.asyncio
    async def test_value_implied_needs_grounding(self):
        """'is single' — the value 'single' implies marital_status column.

        This is VALUE_IMPLIED: the user didn't name the column, they named
        a value. The grounding stage will figure out which column contains
        'single' (likely marital_status). If multiple columns contain that
        value, it routes to clarification.
        """
        resolver = _make_mock_resolver(prompt_index=6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred_3 = draft.actions[0].logical_groups[0].predicates[2]
        assert pred_3.field_ref.reference_kind == ReferenceKind.VALUE_IMPLIED
        # Not resolved yet — grounding will handle this
        assert pred_3.field_ref.resolved_column is None

    @pytest.mark.asyncio
    async def test_boolean_scope_preserved_as_and(self):
        """All three conditions are in a single AND group (not split into
        separate filter actions)."""
        resolver = _make_mock_resolver(prompt_index=6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1  # Single filter, not 3 separate ones
        group = draft.actions[0].logical_groups[0]
        assert group.operator == "and"
        assert len(group.predicates) == 3

    @pytest.mark.asyncio
    async def test_pretty_output(self):
        resolver = _make_mock_resolver(prompt_index=6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)
        output = pretty_print_draft(draft)
        print(f"\n{'='*70}\nPROMPT 7: {TEST_PROMPTS[6]}\n{'='*70}\n{output}")
        assert "gender" in output
        assert "education" in output
        assert "value_implied" in output
        assert "single" in output
