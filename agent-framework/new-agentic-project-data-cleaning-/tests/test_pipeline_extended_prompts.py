"""Extended pipeline tests: 14 additional prompts through SemanticExtractor.

Tests the new SemanticExtractor pipeline with mocked LLM responses to validate
correct SemanticIntentDraft output for various prompt patterns including:
- Value-list filters (operator="in")
- Drop actions with explicit column names
- Semantic concept references (multi-word → single column)
- Comparison operators (gt for "above than")
- Multi-predicate AND groups
- Value-implied references

Each test validates action type, reference_kind, operator, values, boolean scope,
and provenance on every element.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest

from finflow_agent.grounding.llm_adapter import LLMCallSite, LLMResponse
from finflow_agent.grounding.semantic_extractor import SemanticExtractor, SchemaContext
from finflow_agent.models.draft import ReferenceKind, ResolutionStatus, SemanticIntentDraft
from finflow_agent.models.pretty_printer import pretty_print_draft


# ---------------------------------------------------------------------------
# Test prompts (14 new prompts)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "Clean the data and extract rows which contains paypal or cash as payment method",
    "Clean the data and extract all columns except Customer_ID",
    "Clean the data and extract all which has transaction status as pending",
    "Clean the data and extract all which has transaction status as completed",
    "Clean the data and extract all which has payment method as credit card",
    "Clean the data and extract all which has price above than 500",
    "Clean the data and extract all columns except loan_amount",
    "Clean the data and extract all which has age above than 45",
    "Clean the data and extract all which has gender as female",
    "Clean the data and extract all which has education as phd",
    "Clean the data and extract all which has loan status as approved",
    "Clean the data and extract all which has gender as female, education as phd and is single",
    "Clean the data and extract all which has employment type as unemployed",
    "Clean the data and extract all which has home ownership as rent",
]

# ---------------------------------------------------------------------------
# Schema context (extended for all 14 prompts)
# ---------------------------------------------------------------------------

SAMPLE_COLUMNS = [
    "consumer_id", "Customer_ID", "age", "gender", "education",
    "marital_status", "payment_method", "payment_status",
    "transaction_status", "amount", "price", "loan_amount",
    "loan_status", "employment_type", "home_ownership", "date",
]

SAMPLE_SCHEMA_CONTEXT = SchemaContext(
    column_names=SAMPLE_COLUMNS,
    column_types={
        "consumer_id": "int64",
        "Customer_ID": "int64",
        "age": "int64",
        "gender": "object",
        "education": "object",
        "marital_status": "object",
        "payment_method": "object",
        "payment_status": "object",
        "transaction_status": "object",
        "amount": "float64",
        "price": "float64",
        "loan_amount": "float64",
        "loan_status": "object",
        "employment_type": "object",
        "home_ownership": "object",
        "date": "datetime64",
    },
    dataset_description="Consumer financial and loan dataset",
)


# ---------------------------------------------------------------------------
# Mock LLM extraction responses for each prompt
# ---------------------------------------------------------------------------


def _make_extraction_response_1() -> str:
    """Prompt 1: 'Clean the data and extract rows which contains paypal or cash as payment method'

    Filter action: field_ref='payment method' (semantic_concept), operator='in',
    value=['paypal','cash']. Boolean scope preserved as single predicate with value list.
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
                                    "reference_text": "payment method",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 64,
                                            "end_offset": 78,
                                            "source_text": "payment method",
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
                                        "end_offset": 78,
                                        "source_text": "contains paypal or cash as payment method",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 78,
                                "source_text": "Clean the data and extract rows which contains paypal or cash as payment method",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 78,
                        "source_text": "extract rows which contains paypal or cash as payment method",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_2() -> str:
    """Prompt 2: 'Clean the data and extract all columns except Customer_ID'

    Drop action: column='Customer_ID' (explicit_name).
    """
    return json.dumps({
        "actions": [
            {
                "type": "drop",
                "columns": [
                    {
                        "reference_text": "Customer_ID",
                        "reference_kind": "explicit_name",
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 49,
                                "end_offset": 60,
                                "source_text": "Customer_ID",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 60,
                        "source_text": "extract all columns except Customer_ID",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_3() -> str:
    """Prompt 3: 'Clean the data and extract all which has transaction status as pending'

    Filter: field_ref='transaction status' (semantic_concept), operator='eq', value='pending'.
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
                                    "reference_text": "transaction status",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 57,
                                            "source_text": "transaction status",
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
                                        "end_offset": 68,
                                        "source_text": "transaction status as pending",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 68,
                                "source_text": "Clean the data and extract all which has transaction status as pending",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 68,
                        "source_text": "extract all which has transaction status as pending",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_4() -> str:
    """Prompt 4: 'Clean the data and extract all which has transaction status as completed'

    Filter: field_ref='transaction status' (semantic_concept), operator='eq', value='completed'.
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
                                    "reference_text": "transaction status",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 57,
                                            "source_text": "transaction status",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "completed",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 70,
                                        "source_text": "transaction status as completed",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 70,
                                "source_text": "Clean the data and extract all which has transaction status as completed",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 70,
                        "source_text": "extract all which has transaction status as completed",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_5() -> str:
    """Prompt 5: 'Clean the data and extract all which has payment method as credit card'

    Filter: field_ref='payment method' (semantic_concept), operator='eq', value='credit card'.
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
                                    "reference_text": "payment method",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 53,
                                            "source_text": "payment method",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "credit card",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 68,
                                        "source_text": "payment method as credit card",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 68,
                                "source_text": "Clean the data and extract all which has payment method as credit card",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 68,
                        "source_text": "extract all which has payment method as credit card",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_6() -> str:
    """Prompt 6: 'Clean the data and extract all which has price above than 500'

    Filter: field_ref='price' (explicit_name), operator='gt', value=500.
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
                                    "reference_text": "price",
                                    "reference_kind": "explicit_name",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 44,
                                            "source_text": "price",
                                        }
                                    ],
                                },
                                "operator": "gt",
                                "value": 500,
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 60,
                                        "source_text": "price above than 500",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 60,
                                "source_text": "Clean the data and extract all which has price above than 500",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 60,
                        "source_text": "extract all which has price above than 500",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_7() -> str:
    """Prompt 7: 'Clean the data and extract all columns except loan_amount'

    Drop action: column='loan_amount' (explicit_name).
    """
    return json.dumps({
        "actions": [
            {
                "type": "drop",
                "columns": [
                    {
                        "reference_text": "loan_amount",
                        "reference_kind": "explicit_name",
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 49,
                                "end_offset": 60,
                                "source_text": "loan_amount",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 60,
                        "source_text": "extract all columns except loan_amount",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_8() -> str:
    """Prompt 8: 'Clean the data and extract all which has age above than 45'

    Filter: field_ref='age' (explicit_name), operator='gt', value=45.
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


def _make_extraction_response_9() -> str:
    """Prompt 9: 'Clean the data and extract all which has gender as female'

    Filter: field_ref='gender' (explicit_name), operator='eq', value='female'.
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


def _make_extraction_response_10() -> str:
    """Prompt 10: 'Clean the data and extract all which has education as phd'

    Filter: field_ref='education' (explicit_name), operator='eq', value='phd'.
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


def _make_extraction_response_11() -> str:
    """Prompt 11: 'Clean the data and extract all which has loan status as approved'

    Filter: field_ref='loan status' (semantic_concept), operator='eq', value='approved'.
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
                                    "reference_text": "loan status",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 50,
                                            "source_text": "loan status",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "approved",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 62,
                                        "source_text": "loan status as approved",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 62,
                                "source_text": "Clean the data and extract all which has loan status as approved",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 62,
                        "source_text": "extract all which has loan status as approved",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_12() -> str:
    """Prompt 12: 'Clean the data and extract all which has gender as female, education as phd and is single'

    Multi-condition filter: 3 predicates in AND group:
    - gender (explicit_name) == 'female'
    - education (explicit_name) == 'phd'
    - 'single' (value_implied) == 'single'
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
                                            "start_offset": 57,
                                            "end_offset": 66,
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
                                        "start_offset": 57,
                                        "end_offset": 73,
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
                                            "start_offset": 81,
                                            "end_offset": 87,
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
                                        "start_offset": 78,
                                        "end_offset": 87,
                                        "source_text": "is single",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 87,
                                "source_text": "Clean the data and extract all which has gender as female, education as phd and is single",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 87,
                        "source_text": "extract all which has gender as female, education as phd and is single",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_13() -> str:
    """Prompt 13: 'Clean the data and extract all which has employment type as unemployed'

    Filter: field_ref='employment type' (semantic_concept), operator='eq', value='unemployed'.
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
                                    "reference_text": "employment type",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 54,
                                            "source_text": "employment type",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "unemployed",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 68,
                                        "source_text": "employment type as unemployed",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 68,
                                "source_text": "Clean the data and extract all which has employment type as unemployed",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 68,
                        "source_text": "extract all which has employment type as unemployed",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


def _make_extraction_response_14() -> str:
    """Prompt 14: 'Clean the data and extract all which has home ownership as rent'

    Filter: field_ref='home ownership' (semantic_concept), operator='eq', value='rent'.
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
                                    "reference_text": "home ownership",
                                    "reference_kind": "semantic_concept",
                                    "provenance": [
                                        {
                                            "type": "prompt_span",
                                            "start_offset": 39,
                                            "end_offset": 53,
                                            "source_text": "home ownership",
                                        }
                                    ],
                                },
                                "operator": "eq",
                                "value": "rent",
                                "negated": False,
                                "provenance": [
                                    {
                                        "type": "prompt_span",
                                        "start_offset": 39,
                                        "end_offset": 61,
                                        "source_text": "home ownership as rent",
                                    }
                                ],
                            }
                        ],
                        "provenance": [
                            {
                                "type": "prompt_span",
                                "start_offset": 0,
                                "end_offset": 61,
                                "source_text": "Clean the data and extract all which has home ownership as rent",
                            }
                        ],
                    }
                ],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 19,
                        "end_offset": 61,
                        "source_text": "extract all which has home ownership as rent",
                    }
                ],
            }
        ],
        "ambiguities": [],
        "ignored_spans": [],
    })


# ---------------------------------------------------------------------------
# Response registry and mock helper
# ---------------------------------------------------------------------------

_EXTRACTION_RESPONSES = [
    _make_extraction_response_1,
    _make_extraction_response_2,
    _make_extraction_response_3,
    _make_extraction_response_4,
    _make_extraction_response_5,
    _make_extraction_response_6,
    _make_extraction_response_7,
    _make_extraction_response_8,
    _make_extraction_response_9,
    _make_extraction_response_10,
    _make_extraction_response_11,
    _make_extraction_response_12,
    _make_extraction_response_13,
    _make_extraction_response_14,
]


def _make_mock_resolver(prompt_index: int) -> AsyncMock:
    """Create a mock resolver that returns the appropriate extraction response."""
    resolver = AsyncMock()
    content = _EXTRACTION_RESPONSES[prompt_index]()
    resolver.call.return_value = LLMResponse(
        content=content,
        parsed=None,
        call_site=LLMCallSite.EXTRACTION,
        latency_ms=150.0,
    )
    return resolver


# ---------------------------------------------------------------------------
# Test classes — one per prompt
# ---------------------------------------------------------------------------


class TestPrompt1PaypalOrCashPaymentMethod:
    """Prompt 1: 'Clean the data and extract rows which contains paypal or cash as payment method'

    Tests: filter action, semantic_concept reference, 'in' operator with value list,
    boolean scope preserved (paypal or cash as single predicate).
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(0)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[0], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert isinstance(draft, SemanticIntentDraft)
        assert len(draft.actions) == 1
        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_reference(self):
        """'payment method' is a semantic_concept (two words → payment_method column)."""
        resolver = _make_mock_resolver(0)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[0], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "payment method"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT

    @pytest.mark.asyncio
    async def test_in_operator_with_value_list(self):
        """'paypal or cash' preserved as single predicate with operator='in' and value list."""
        resolver = _make_mock_resolver(0)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[0], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.operator == "in"
        assert pred.value == ["paypal", "cash"]

    @pytest.mark.asyncio
    async def test_boolean_scope_preserved(self):
        """'paypal or cash' is ONE predicate, NOT two separate filter clauses."""
        resolver = _make_mock_resolver(0)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[0], schema_context=SAMPLE_SCHEMA_CONTEXT)

        predicates = draft.actions[0].logical_groups[0].predicates
        assert len(predicates) == 1  # Single predicate with value list

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(0)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[0], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1
        for action in draft.actions:
            assert len(action.provenance) >= 1


class TestPrompt2DropCustomerID:
    """Prompt 2: 'Clean the data and extract all columns except Customer_ID'

    Tests: drop action, explicit_name reference for exact column name.
    """

    @pytest.mark.asyncio
    async def test_produces_drop_action(self):
        resolver = _make_mock_resolver(1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1
        assert draft.actions[0].type == "drop"

    @pytest.mark.asyncio
    async def test_explicit_name_reference(self):
        """'Customer_ID' matches exact column name → explicit_name."""
        resolver = _make_mock_resolver(1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)

        col_ref = draft.actions[0].columns[0]
        assert col_ref.reference_text == "Customer_ID"
        assert col_ref.reference_kind == ReferenceKind.EXPLICIT_NAME

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(1)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[1], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1
        assert len(draft.actions[0].provenance) >= 1
        assert len(draft.actions[0].columns[0].provenance) >= 1


class TestPrompt3TransactionStatusPending:
    """Prompt 3: 'Clean the data and extract all which has transaction status as pending'

    Tests: filter, semantic_concept reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_reference(self):
        """'transaction status' is a semantic_concept (two words → transaction_status)."""
        resolver = _make_mock_resolver(2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "transaction status"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT

    @pytest.mark.asyncio
    async def test_eq_operator_and_value(self):
        resolver = _make_mock_resolver(2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.operator == "eq"
        assert pred.value == "pending"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(2)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[2], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert len(pred.provenance) >= 1
        assert len(pred.field_ref.provenance) >= 1


class TestPrompt4TransactionStatusCompleted:
    """Prompt 4: 'Clean the data and extract all which has transaction status as completed'

    Tests: filter, semantic_concept reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(3)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[3], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_and_value(self):
        resolver = _make_mock_resolver(3)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[3], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "transaction status"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "completed"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(3)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[3], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1


class TestPrompt5PaymentMethodCreditCard:
    """Prompt 5: 'Clean the data and extract all which has payment method as credit card'

    Tests: filter, semantic_concept reference, eq operator with multi-word value.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(4)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[4], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_and_value(self):
        resolver = _make_mock_resolver(4)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[4], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "payment method"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "credit card"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(4)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[4], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert len(pred.provenance) >= 1
        assert len(pred.field_ref.provenance) >= 1


class TestPrompt6PriceAbove500:
    """Prompt 6: 'Clean the data and extract all which has price above than 500'

    Tests: filter, explicit_name reference, gt operator, numeric value.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(5)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[5], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_explicit_name_gt_operator(self):
        resolver = _make_mock_resolver(5)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[5], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "price"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "gt"
        assert pred.value == 500

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(5)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[5], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1


class TestPrompt7DropLoanAmount:
    """Prompt 7: 'Clean the data and extract all columns except loan_amount'

    Tests: drop action, explicit_name reference.
    """

    @pytest.mark.asyncio
    async def test_produces_drop_action(self):
        resolver = _make_mock_resolver(6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1
        assert draft.actions[0].type == "drop"

    @pytest.mark.asyncio
    async def test_explicit_name_reference(self):
        resolver = _make_mock_resolver(6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        col_ref = draft.actions[0].columns[0]
        assert col_ref.reference_text == "loan_amount"
        assert col_ref.reference_kind == ReferenceKind.EXPLICIT_NAME

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(6)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[6], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions[0].provenance) >= 1
        assert len(draft.actions[0].columns[0].provenance) >= 1


class TestPrompt8AgeAbove45:
    """Prompt 8: 'Clean the data and extract all which has age above than 45'

    Tests: filter, explicit_name reference, gt operator, numeric value.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_with_gt(self):
        resolver = _make_mock_resolver(7)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[7], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "age"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "gt"
        assert pred.value == 45

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(7)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[7], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1


class TestPrompt9GenderFemale:
    """Prompt 9: 'Clean the data and extract all which has gender as female'

    Tests: filter, explicit_name reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_with_eq(self):
        resolver = _make_mock_resolver(8)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[8], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "gender"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "eq"
        assert pred.value == "female"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(8)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[8], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1


class TestPrompt10EducationPhd:
    """Prompt 10: 'Clean the data and extract all which has education as phd'

    Tests: filter, explicit_name reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_with_eq(self):
        resolver = _make_mock_resolver(9)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[9], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "education"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "eq"
        assert pred.value == "phd"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(9)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[9], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1


class TestPrompt11LoanStatusApproved:
    """Prompt 11: 'Clean the data and extract all which has loan status as approved'

    Tests: filter, semantic_concept reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(10)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[10], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_and_value(self):
        resolver = _make_mock_resolver(10)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[10], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "loan status"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "approved"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(10)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[10], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert len(pred.provenance) >= 1
        assert len(pred.field_ref.provenance) >= 1


class TestPrompt12MultiConditionFemalePhDSingle:
    """Prompt 12: 'Clean the data and extract all which has gender as female, education as phd and is single'

    Tests: filter with 3 predicates in AND group, mixed reference kinds,
    value_implied for 'single'.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.actions) == 1
        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_three_predicates_in_and_group(self):
        """Boolean scope preserved: 3 conditions in one AND group, not 3 separate filters."""
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        group = draft.actions[0].logical_groups[0]
        assert group.operator == "and"
        assert len(group.predicates) == 3

    @pytest.mark.asyncio
    async def test_predicate_1_gender_female(self):
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "gender"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "eq"
        assert pred.value == "female"

    @pytest.mark.asyncio
    async def test_predicate_2_education_phd(self):
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[1]
        assert pred.field_ref.reference_text == "education"
        assert pred.field_ref.reference_kind == ReferenceKind.EXPLICIT_NAME
        assert pred.operator == "eq"
        assert pred.value == "phd"

    @pytest.mark.asyncio
    async def test_predicate_3_value_implied_single(self):
        """'is single' uses value_implied — the value implies the column (marital_status)."""
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[2]
        assert pred.field_ref.reference_text == "single"
        assert pred.field_ref.reference_kind == ReferenceKind.VALUE_IMPLIED
        assert pred.operator == "eq"
        assert pred.value == "single"

    @pytest.mark.asyncio
    async def test_provenance_exists_on_all_predicates(self):
        resolver = _make_mock_resolver(11)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[11], schema_context=SAMPLE_SCHEMA_CONTEXT)

        for pred in draft.actions[0].logical_groups[0].predicates:
            assert len(pred.provenance) >= 1
            assert len(pred.field_ref.provenance) >= 1


class TestPrompt13EmploymentTypeUnemployed:
    """Prompt 13: 'Clean the data and extract all which has employment type as unemployed'

    Tests: filter, semantic_concept reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(12)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[12], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_and_value(self):
        resolver = _make_mock_resolver(12)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[12], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "employment type"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "unemployed"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(12)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[12], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1
        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert len(pred.provenance) >= 1


class TestPrompt14HomeOwnershipRent:
    """Prompt 14: 'Clean the data and extract all which has home ownership as rent'

    Tests: filter, semantic_concept reference, eq operator.
    """

    @pytest.mark.asyncio
    async def test_produces_filter_action(self):
        resolver = _make_mock_resolver(13)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[13], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert draft.actions[0].type == "filter"

    @pytest.mark.asyncio
    async def test_semantic_concept_and_value(self):
        resolver = _make_mock_resolver(13)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[13], schema_context=SAMPLE_SCHEMA_CONTEXT)

        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert pred.field_ref.reference_text == "home ownership"
        assert pred.field_ref.reference_kind == ReferenceKind.SEMANTIC_CONCEPT
        assert pred.operator == "eq"
        assert pred.value == "rent"

    @pytest.mark.asyncio
    async def test_provenance_exists(self):
        resolver = _make_mock_resolver(13)
        extractor = SemanticExtractor(resolver)
        draft = await extractor.extract(TEST_PROMPTS[13], schema_context=SAMPLE_SCHEMA_CONTEXT)

        assert len(draft.extraction_provenance) >= 1
        pred = draft.actions[0].logical_groups[0].predicates[0]
        assert len(pred.provenance) >= 1
        assert len(pred.field_ref.provenance) >= 1
