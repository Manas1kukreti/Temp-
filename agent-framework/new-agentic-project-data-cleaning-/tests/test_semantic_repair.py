"""Unit tests for the Semantic Repair module.

Tests the bounded repair contract:
- Maximum one attempt per pipeline invocation (Req 4.2)
- Only accepts declared patch paths from validator failures (Req 4.1)
- Returns typed SemanticPatch list (add/replace/remove only) (Req 4.3)
- Logging of patch activity (Req 4.6)
- Reset for new invocations
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

import pytest

from finflow_agent.grounding.llm_adapter import (
    LLMCallSite,
    LLMConstraint,
    LLMResponse,
    SemanticResolver,
)
from finflow_agent.models.draft import (
    DropAction,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    ReferenceKind,
)
from finflow_agent.models.patches import PatchOp, SemanticPatch
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.pipeline.coverage_validator import (
    FailureCategory,
    StructuralFailure,
)
from finflow_agent.pipeline.semantic_repair import (
    RepairAlreadyAttemptedError,
    SemanticRepair,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_draft() -> SemanticIntentDraft:
    """Create a minimal valid SemanticIntentDraft for testing."""
    prov = PromptSpanProvenance(
        type="prompt_span",
        start_offset=0,
        end_offset=40,
        source_text="remove duplicates from payment_method",
    )
    col_ref = SemanticColumnReference(
        reference_text="payment_method",
        reference_kind=ReferenceKind.EXPLICIT_NAME,
        provenance=[prov],
    )
    action = DropAction(
        columns=[col_ref],
        provenance=[prov],
    )
    return SemanticIntentDraft(
        raw_prompt="remove duplicates from payment_method",
        actions=[action],
        resolution_status=ResolutionStatus.PENDING,
    )


def _make_failures() -> list[StructuralFailure]:
    """Create sample structural failures."""
    return [
        StructuralFailure(
            category=FailureCategory.PROVENANCE_MISSING,
            element_path="actions[0].provenance",
            description="Action at index 0 has no provenance references",
        ),
        StructuralFailure(
            category=FailureCategory.ACTION_REFERENCE_INCOMPLETE,
            element_path="actions[0].column_refs[0]",
            description="Column reference 'payment_method' not linked to action",
        ),
    ]


def _make_resolver_mock(
    parsed_response: dict[str, Any] | None = None,
) -> AsyncMock:
    """Create a mock SemanticResolver that returns a configured response.

    Note: LLMResponse.parsed is dict | None (strict Pydantic). Lists must be
    wrapped in a dict (e.g., {"patches": [...]}).
    """
    mock = AsyncMock(spec=SemanticResolver)
    mock.call = AsyncMock(
        return_value=LLMResponse(
            content="{}",
            parsed=parsed_response,
            call_site=LLMCallSite.REPAIR,
            latency_ms=150.0,
            retries_used=0,
        )
    )
    return mock


# ---------------------------------------------------------------------------
# Tests: Maximum one attempt per invocation (Req 4.2)
# ---------------------------------------------------------------------------


class TestMaxOneAttempt:
    """Verify the bounded repair contract of one attempt per invocation."""

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self):
        """First repair attempt should succeed without error."""
        resolver = _make_resolver_mock(parsed_response={"patches": []})
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), _make_failures(), "test prompt")

        assert isinstance(result, list)
        assert repair.attempted is True

    @pytest.mark.asyncio
    async def test_second_attempt_raises_error(self):
        """Second repair attempt in same invocation should raise."""
        resolver = _make_resolver_mock(parsed_response={"patches": []})
        repair = SemanticRepair(resolver)

        # First attempt
        await repair.repair(_make_draft(), _make_failures(), "test prompt")

        # Second attempt should raise
        with pytest.raises(RepairAlreadyAttemptedError):
            await repair.repair(_make_draft(), _make_failures(), "test prompt")

    @pytest.mark.asyncio
    async def test_reset_allows_new_attempt(self):
        """After reset, a new repair attempt should be allowed."""
        resolver = _make_resolver_mock(parsed_response={"patches": []})
        repair = SemanticRepair(resolver)

        await repair.repair(_make_draft(), _make_failures(), "test prompt")
        assert repair.attempted is True

        repair.reset()
        assert repair.attempted is False

        # Should succeed after reset
        result = await repair.repair(_make_draft(), _make_failures(), "test prompt")
        assert isinstance(result, list)

    def test_initial_state_not_attempted(self):
        """Fresh instance should not be in attempted state."""
        resolver = _make_resolver_mock()
        repair = SemanticRepair(resolver)
        assert repair.attempted is False


# ---------------------------------------------------------------------------
# Tests: Only declared paths accepted (Req 4.1)
# ---------------------------------------------------------------------------


class TestDeclaredPathsOnly:
    """Verify only declared patch paths from validator failures are accepted."""

    @pytest.mark.asyncio
    async def test_rejects_undeclared_path(self):
        """Patches targeting undeclared paths should be filtered out."""
        # LLM returns a patch for an undeclared path
        parsed = {"patches": [
            {
                "operation": "add",
                "path": "actions[99].something_undeclared",
                "value": "test",
                "reason": "trying to patch wrong path",
                "source_failure": "provenance_missing",
            }
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), _make_failures(), "test prompt")

        # Patch should be filtered out since path isn't in declared failures
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_accepts_declared_path(self):
        """Patches targeting declared paths should be included."""
        failures = _make_failures()
        declared_path = failures[0].element_path  # "actions[0].provenance"

        parsed = {"patches": [
            {
                "operation": "add",
                "path": declared_path,
                "value": {"type": "prompt_span", "start": 0, "end": 5, "text": "remove"},
                "reason": "Adding missing provenance",
                "source_failure": "provenance_missing",
            }
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")

        assert len(result) == 1
        assert result[0].path == declared_path
        assert result[0].operation == PatchOp.ADD


# ---------------------------------------------------------------------------
# Tests: Typed patches only (Req 4.3)
# ---------------------------------------------------------------------------


class TestTypedPatchesOnly:
    """Verify only add/replace/remove operations are returned."""

    @pytest.mark.asyncio
    async def test_rejects_invalid_operation(self):
        """Patches with invalid operation types should be filtered."""
        failures = _make_failures()
        parsed = {"patches": [
            {
                "operation": "rewrite",  # Not a valid PatchOp
                "path": failures[0].element_path,
                "value": "something",
                "reason": "invalid op",
            }
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_accepts_all_valid_operations(self):
        """All three valid operations (add, replace, remove) should be accepted."""
        failures = [
            StructuralFailure(
                category=FailureCategory.PROVENANCE_MISSING,
                element_path="actions[0].provenance",
                description="Missing provenance",
            ),
            StructuralFailure(
                category=FailureCategory.ACTION_REFERENCE_INCOMPLETE,
                element_path="actions[0].column_refs[0]",
                description="Incomplete reference",
            ),
            StructuralFailure(
                category=FailureCategory.DUPLICATE_ACTION,
                element_path="actions[1]",
                description="Duplicate action",
            ),
        ]
        parsed = {"patches": [
            {
                "operation": "add",
                "path": "actions[0].provenance",
                "value": {"type": "prompt_span"},
                "reason": "Add provenance",
                "source_failure": "provenance_missing",
            },
            {
                "operation": "replace",
                "path": "actions[0].column_refs[0]",
                "value": {"name": "payment_method"},
                "reason": "Replace ref",
                "source_failure": "action_reference_incomplete",
            },
            {
                "operation": "remove",
                "path": "actions[1]",
                "reason": "Remove duplicate",
                "source_failure": "duplicate_action",
            },
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")

        assert len(result) == 3
        ops = {p.operation for p in result}
        assert ops == {PatchOp.ADD, PatchOp.REPLACE, PatchOp.REMOVE}

    @pytest.mark.asyncio
    async def test_remove_patch_has_no_value(self):
        """Remove patches should have value=None."""
        failures = [
            StructuralFailure(
                category=FailureCategory.DUPLICATE_ACTION,
                element_path="actions[1]",
                description="Duplicate action",
            ),
        ]
        parsed = {"patches": [
            {
                "operation": "remove",
                "path": "actions[1]",
                "value": "should_be_ignored",
                "reason": "Remove duplicate",
                "source_failure": "duplicate_action",
            },
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")

        assert len(result) == 1
        assert result[0].operation == PatchOp.REMOVE
        assert result[0].value is None

    @pytest.mark.asyncio
    async def test_add_without_value_rejected(self):
        """Add patches without a value should be rejected."""
        failures = _make_failures()
        parsed = {"patches": [
            {
                "operation": "add",
                "path": failures[0].element_path,
                "reason": "Add without value",
                "source_failure": "provenance_missing",
                # Missing "value" field
            },
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Empty failures and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and degenerate inputs."""

    @pytest.mark.asyncio
    async def test_empty_failures_returns_empty_patches(self):
        """With no failures, should return empty list without calling LLM."""
        resolver = _make_resolver_mock()
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), [], "test prompt")

        assert result == []
        # LLM should NOT be called when there are no failures
        resolver.call.assert_not_called()
        # But attempt should still be marked
        assert repair.attempted is True

    @pytest.mark.asyncio
    async def test_none_parsed_response_returns_empty(self):
        """If LLM returns None parsed content, return empty list."""
        resolver = _make_resolver_mock(parsed_response=None)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), _make_failures(), "test prompt")
        assert result == []

    @pytest.mark.asyncio
    async def test_dict_with_patches_key(self):
        """Response as dict with 'patches' key should be parsed."""
        failures = _make_failures()
        parsed = {
            "patches": [
                {
                    "operation": "add",
                    "path": failures[0].element_path,
                    "value": {"type": "prompt_span"},
                    "reason": "Fix",
                    "source_failure": "provenance_missing",
                }
            ]
        }
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_result_types_are_semantic_patch(self):
        """All returned items must be SemanticPatch instances."""
        failures = _make_failures()
        parsed = {"patches": [
            {
                "operation": "replace",
                "path": failures[0].element_path,
                "value": "new_value",
                "reason": "Fix provenance",
                "source_failure": "provenance_missing",
            },
        ]}
        resolver = _make_resolver_mock(parsed_response=parsed)
        repair = SemanticRepair(resolver)

        result = await repair.repair(_make_draft(), failures, "test prompt")

        for patch in result:
            assert isinstance(patch, SemanticPatch)
