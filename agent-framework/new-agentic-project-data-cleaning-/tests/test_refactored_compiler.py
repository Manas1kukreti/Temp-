"""Tests for the refactored Compiler contract (semantic-grounding-refactor spec).

Validates:
- Compiler accepts only CanonicalIntent objects (Req 11.2)
- Compiler validates every column reference is resolved (Req 11.1)
- Compiler does NOT perform grounding, LLM calls, or semantic re-interpretation (Req 6.4)
- Compiler produces a RefactoredExecutionPlan (Req 6.1)

Note: Due to a pre-existing circular import in the legacy codebase
(finflow_agent.state <-> finflow_agent.execution), we import the new
Compiler code via a targeted import that avoids triggering the legacy
import chain through visualization_agent.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Ensure src is on path
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import new models (these don't trigger the circular import)
from finflow_agent.models.canonical import (
    CanonicalIntent,
    ResolvedDropAction,
    ResolvedFilterAction,
    ResolvedProjectAction,
    ResolvedRenameAction,
    ResolvedSortAction,
)
from finflow_agent.models.draft import ResolutionOrigin
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.models.snapshot import DataSnapshotRef


def _load_compiler_module():
    """Load the compiler module in a way that handles the pre-existing circular import.

    The legacy compiler.py imports from finflow_agent.agents.visualization_agent
    which triggers a circular import chain. We handle this by attempting the import
    and if the circular import is hit on Python 3.14, we fall back to a workaround.
    """
    try:
        from finflow_agent.planning.compiler import (
            ActionType,
            Compiler,
            CompilerError,
            ExecutionStep,
            RefactoredExecutionPlan,
        )
        return ActionType, Compiler, CompilerError, ExecutionStep, RefactoredExecutionPlan
    except ImportError:
        # Circular import in legacy code — mock out the problematic imports
        # and retry. This is safe because we only test the NEW code path.
        import unittest.mock

        # Create minimal mocks for the circular import chain
        if "finflow_agent.state" not in sys.modules:
            # Pre-populate the module to break the cycle
            state_spec = importlib.util.spec_from_file_location(
                "finflow_agent.execution.state",
                os.path.join(SRC_DIR, "finflow_agent", "execution", "state.py"),
            )
            state_mod = importlib.util.module_from_spec(state_spec)
            sys.modules["finflow_agent.execution.state"] = state_mod
            state_spec.loader.exec_module(state_mod)

            # Now populate finflow_agent.state
            state_facade_spec = importlib.util.spec_from_file_location(
                "finflow_agent.state",
                os.path.join(SRC_DIR, "finflow_agent", "state.py"),
            )
            state_facade_mod = importlib.util.module_from_spec(state_facade_spec)
            sys.modules["finflow_agent.state"] = state_facade_mod
            state_facade_spec.loader.exec_module(state_facade_mod)

        # Mock visualization_agent to avoid the chain
        mock_vis = unittest.mock.MagicMock()
        mock_vis.VISUALIZATION_DISABLED_MESSAGE = "visualization is disabled"
        sys.modules.setdefault("finflow_agent.agents.visualization_agent", mock_vis)

        # Mock contract_registry
        mock_cr = unittest.mock.MagicMock()
        sys.modules.setdefault("finflow_agent.contract_registry", mock_cr)

        # Mock operations.schemas
        mock_ops = unittest.mock.MagicMock()
        sys.modules.setdefault("finflow_agent.operations.schemas", mock_ops)

        # Mock planning.canonical_intent
        mock_ci = unittest.mock.MagicMock()
        mock_ci.CANONICAL_INTENT_SCHEMA_VERSION = "1.0"
        sys.modules.setdefault("finflow_agent.planning.canonical_intent", mock_ci)

        # Mock planning.intent_schema
        mock_is = unittest.mock.MagicMock()
        sys.modules.setdefault("finflow_agent.planning.intent_schema", mock_is)

        # Mock tools.config
        mock_tc = unittest.mock.MagicMock()
        sys.modules.setdefault("finflow_agent.tools.config", mock_tc)

        from finflow_agent.planning.compiler import (
            ActionType,
            Compiler,
            CompilerError,
            ExecutionStep,
            RefactoredExecutionPlan,
        )
        return ActionType, Compiler, CompilerError, ExecutionStep, RefactoredExecutionPlan


ActionType, Compiler, CompilerError, ExecutionStep, RefactoredExecutionPlan = _load_compiler_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_provenance():
    return [
        PromptSpanProvenance(
            start_offset=0, end_offset=10, source_text="test prompt"
        )
    ]


def _make_snapshot():
    return DataSnapshotRef(
        file_id="file-001",
        content_hash="sha256:abc123",
        byte_size=1024,
        storage_version="v1",
        profile_id="profile-001",
        structural_schema_fingerprint="fp-struct-001",
        profile_fingerprint="fp-profile-001",
    )


def _make_canonical_intent(actions):
    """Helper to build a valid CanonicalIntent for testing."""
    return CanonicalIntent(
        resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
        actions=actions,
        source_draft_id="draft-001",
        source_draft_revision=3,
        data_snapshot_ref=_make_snapshot(),
        provenance=_make_provenance(),
    )


# ---------------------------------------------------------------------------
# Tests: Type-level guarantee (Req 11.2)
# ---------------------------------------------------------------------------


class TestCompilerTypeContract:
    """Compiler must only accept CanonicalIntent objects."""

    def test_rejects_non_canonical_intent(self):
        compiler = Compiler()
        with pytest.raises(CompilerError, match="Compiler accepts only CanonicalIntent"):
            compiler.compile("not a canonical intent")  # type: ignore

    def test_rejects_dict_input(self):
        compiler = Compiler()
        with pytest.raises(CompilerError, match="Compiler accepts only CanonicalIntent"):
            compiler.compile({"actions": []})  # type: ignore

    def test_rejects_none_input(self):
        compiler = Compiler()
        with pytest.raises(CompilerError, match="Compiler accepts only CanonicalIntent"):
            compiler.compile(None)  # type: ignore

    def test_rejects_empty_actions(self):
        """CanonicalIntent with no actions should be rejected."""
        compiler = Compiler()
        intent = CanonicalIntent.model_construct(
            intent_id="test-id",
            schema_version="1.0",
            resolution_status="resolved",
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            actions=[],
            source_draft_id="draft-001",
            source_draft_revision=1,
            data_snapshot_ref=_make_snapshot(),
        )
        with pytest.raises(CompilerError, match="no actions"):
            compiler.compile(intent)


# ---------------------------------------------------------------------------
# Tests: Column validation (Req 11.1)
# ---------------------------------------------------------------------------


class TestCompilerColumnValidation:
    """Compiler validates every column reference is resolved."""

    def test_compiles_project_action_with_resolved_columns(self):
        compiler = Compiler()
        action = ResolvedProjectAction(
            columns=["amount", "date", "vendor"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert isinstance(plan, RefactoredExecutionPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].action_type == ActionType.PROJECT
        assert plan.steps[0].resolved_columns == ["amount", "date", "vendor"]

    def test_compiles_drop_action(self):
        compiler = Compiler()
        action = ResolvedDropAction(
            columns=["temp_col", "debug_col"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert len(plan.steps) == 1
        assert plan.steps[0].action_type == ActionType.DROP
        assert plan.steps[0].resolved_columns == ["temp_col", "debug_col"]

    def test_compiles_sort_action(self):
        compiler = Compiler()
        action = ResolvedSortAction(
            keys=["date", "amount"],
            directions=["asc", "desc"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert len(plan.steps) == 1
        assert plan.steps[0].action_type == ActionType.SORT
        assert plan.steps[0].resolved_columns == ["date", "amount"]
        assert plan.steps[0].params["directions"] == ["asc", "desc"]

    def test_compiles_rename_action(self):
        compiler = Compiler()
        action = ResolvedRenameAction(
            mappings=[("old_col", "new_col"), ("amt", "amount")],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert len(plan.steps) == 1
        assert plan.steps[0].action_type == ActionType.RENAME
        assert plan.steps[0].resolved_columns == ["old_col", "amt"]
        assert plan.steps[0].params["rename_map"] == {
            "old_col": "new_col",
            "amt": "amount",
        }

    def test_compiles_filter_action(self):
        compiler = Compiler()
        action = ResolvedFilterAction(
            predicates=[
                {
                    "column": "amount",
                    "operator": "gt",
                    "value": 100,
                    "negated": False,
                    "logical_operator": "and",
                }
            ],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert len(plan.steps) == 1
        assert plan.steps[0].action_type == ActionType.FILTER
        assert plan.steps[0].resolved_columns == ["amount"]

    def test_compiles_multiple_actions_in_order(self):
        compiler = Compiler()
        actions = [
            ResolvedFilterAction(
                predicates=[{"column": "status", "operator": "eq", "value": "active", "negated": False, "logical_operator": "and"}],
                provenance=_make_provenance(),
            ),
            ResolvedProjectAction(
                columns=["name", "status", "amount"],
                provenance=_make_provenance(),
            ),
            ResolvedSortAction(
                keys=["amount"],
                directions=["desc"],
                provenance=_make_provenance(),
            ),
        ]
        intent = _make_canonical_intent(actions)
        plan = compiler.compile(intent)

        assert len(plan.steps) == 3
        assert plan.steps[0].action_type == ActionType.FILTER
        assert plan.steps[1].action_type == ActionType.PROJECT
        assert plan.steps[2].action_type == ActionType.SORT


# ---------------------------------------------------------------------------
# Tests: Execution plan metadata (Req 6.1)
# ---------------------------------------------------------------------------


class TestExecutionPlanMetadata:
    """ExecutionPlan captures intent metadata for traceability."""

    def test_plan_captures_intent_id(self):
        compiler = Compiler()
        action = ResolvedProjectAction(
            columns=["col_a"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert plan.intent_id == intent.intent_id
        assert plan.source_draft_id == "draft-001"
        assert plan.source_draft_revision == 3

    def test_plan_has_unique_id(self):
        compiler = Compiler()
        action = ResolvedProjectAction(
            columns=["col_a"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan1 = compiler.compile(intent)
        plan2 = compiler.compile(intent)

        assert plan1.plan_id != plan2.plan_id

    def test_execution_step_has_unique_id(self):
        compiler = Compiler()
        action = ResolvedProjectAction(
            columns=["col_a"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        assert plan.steps[0].step_id  # non-empty


# ---------------------------------------------------------------------------
# Tests: No grounding, no LLM, no re-interpretation (Req 6.4)
# ---------------------------------------------------------------------------


class TestCompilerNoGrounding:
    """Compiler does not perform grounding, LLM calls, or semantic re-interpretation."""

    def test_compiler_is_synchronous(self):
        """The compile method is NOT async — it does no I/O or LLM calls."""
        compiler = Compiler()
        import inspect

        assert not inspect.iscoroutinefunction(compiler.compile)

    def test_compiler_has_no_llm_dependencies(self):
        """The Compiler class has no LLM-related attributes or dependencies."""
        compiler = Compiler()
        # No attributes related to LLM, grounding, or resolution
        attrs = dir(compiler)
        for attr in attrs:
            if attr.startswith("_"):
                continue
            assert "llm" not in attr.lower()
            assert "grounding" not in attr.lower()
            assert "resolve" not in attr.lower()


# ---------------------------------------------------------------------------
# Tests: ExecutionStep and RefactoredExecutionPlan are frozen (immutable)
# ---------------------------------------------------------------------------


class TestImmutability:
    """Execution models are frozen to prevent mutation after creation."""

    def test_execution_step_is_frozen(self):
        step = ExecutionStep(
            action_type=ActionType.PROJECT,
            resolved_columns=["col_a"],
            params={},
        )
        with pytest.raises(Exception):
            step.action_type = ActionType.FILTER  # type: ignore

    def test_execution_plan_is_frozen(self):
        compiler = Compiler()
        action = ResolvedProjectAction(
            columns=["col_a"],
            provenance=_make_provenance(),
        )
        intent = _make_canonical_intent([action])
        plan = compiler.compile(intent)

        with pytest.raises(Exception):
            plan.intent_id = "tampered"  # type: ignore
