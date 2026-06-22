"""Unit tests for the pipeline observability module.

Tests:
1. PipelineTracingContext - structured tracing fields and timed_stage
2. PipelineMetrics - metric emission and counters for all event types
3. ShadowModeRecorder - shadow mode comparison recording
4. DecisionOwnerRecorder - decision-owner recording per element
5. JSONFormatter - structured JSON log output format

Requirements: 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from finflow_agent.pipeline.observability import (
    DecisionOwnerRecord,
    DecisionOwnerRecorder,
    JSONFormatter,
    MetricEvent,
    PipelineMetrics,
    PipelineStage,
    PipelineTracingContext,
    ShadowModeRecorder,
    get_pipeline_logger,
)
from finflow_agent.models.envelope import ShadowComparisonMetric


# ---------------------------------------------------------------------------
# PipelineTracingContext tests (Requirement 12.2)
# ---------------------------------------------------------------------------


class TestPipelineTracingContext:
    """Tests for the structured tracing context dataclass."""

    def test_all_required_fields_present(self):
        """Tracing context must have all fields from Requirement 12.2."""
        ctx = PipelineTracingContext(submission_id="sub-001")
        d = ctx.to_dict()
        required_fields = [
            "submission_id",
            "draft_id",
            "draft_revision",
            "intent_id",
            "schema_fingerprint",
            "profile_fingerprint",
            "data_snapshot_ref",
            "model_version",
            "pipeline_stage",
            "decision_owner",
            "duration_ms",
        ]
        for f in required_fields:
            assert f in d, f"Missing required tracing field: {f}"

    def test_to_dict_serializes_values(self):
        """to_dict should reflect the values set on the context."""
        ctx = PipelineTracingContext(
            submission_id="sub-x",
            draft_id="draft-1",
            draft_revision=3,
            intent_id="int-42",
            schema_fingerprint="sha256-abc",
            profile_fingerprint="sha256-def",
            data_snapshot_ref="snap-ref-1",
            model_version="v2.1",
            pipeline_stage="extraction",
            decision_owner="SemanticExtractor",
            duration_ms=42.5,
        )
        d = ctx.to_dict()
        assert d["submission_id"] == "sub-x"
        assert d["draft_revision"] == 3
        assert d["duration_ms"] == 42.5

    def test_timed_stage_records_duration(self):
        """timed_stage context manager should measure elapsed time."""
        ctx = PipelineTracingContext(submission_id="s-1")
        with ctx.timed_stage("grounding", "ColumnGrounder") as c:
            time.sleep(0.01)
        assert c.pipeline_stage == "grounding"
        assert c.decision_owner == "ColumnGrounder"
        assert c.duration_ms >= 5.0  # At least ~10ms sleep

    def test_timed_stage_sets_stage_and_owner(self):
        """timed_stage should update pipeline_stage and decision_owner."""
        ctx = PipelineTracingContext(submission_id="s-2")
        with ctx.timed_stage(
            PipelineStage.PREDICATE_GROUNDING.value, "PredicateGrounder"
        ):
            pass
        assert ctx.pipeline_stage == "predicate_grounding"
        assert ctx.decision_owner == "PredicateGrounder"


# ---------------------------------------------------------------------------
# PipelineMetrics tests (Requirement 12.1)
# ---------------------------------------------------------------------------


class TestPipelineMetrics:
    """Tests for metric emission covering all pipeline events."""

    def test_extraction_metrics(self):
        """Extraction attempt/success/failure metrics are tracked."""
        metrics = PipelineMetrics()
        ctx = PipelineTracingContext(submission_id="m-1")
        metrics.record_extraction_attempt(ctx)
        metrics.record_extraction_success(ctx)
        metrics.record_extraction_failure(ctx, reason="timeout")
        assert metrics.get_count(MetricEvent.EXTRACTION_ATTEMPT) == 1
        assert metrics.get_count(MetricEvent.EXTRACTION_SUCCESS) == 1
        assert metrics.get_count(MetricEvent.EXTRACTION_FAILURE) == 1

    def test_grounding_metrics(self):
        """Grounding attempt/success/LLM-fallback metrics are tracked."""
        metrics = PipelineMetrics()
        metrics.record_grounding_attempt()
        metrics.record_grounding_success()
        metrics.record_grounding_llm_fallback()
        assert metrics.get_count(MetricEvent.GROUNDING_ATTEMPT) == 1
        assert metrics.get_count(MetricEvent.GROUNDING_SUCCESS) == 1
        assert metrics.get_count(MetricEvent.GROUNDING_LLM_FALLBACK_INVOCATION) == 1

    def test_post_llm_verification_metrics(self):
        """Post-LLM verification pass/failure metrics are tracked."""
        metrics = PipelineMetrics()
        metrics.record_post_llm_verification_pass()
        metrics.record_post_llm_verification_failure(
            checks_failed=["dtype_compatible"]
        )
        assert metrics.get_count(MetricEvent.POST_LLM_VERIFICATION_PASS) == 1
        assert metrics.get_count(MetricEvent.POST_LLM_VERIFICATION_FAILURE) == 1

    def test_clarification_metrics(self):
        """Clarification initiated/resolved metrics are tracked."""
        metrics = PipelineMetrics()
        metrics.record_clarification_initiated()
        metrics.record_clarification_resolved()
        assert metrics.get_count(MetricEvent.CLARIFICATION_SESSION_INITIATED) == 1
        assert metrics.get_count(MetricEvent.CLARIFICATION_SESSION_RESOLVED) == 1

    def test_repair_metrics(self):
        """Repair attempt/success metrics are tracked."""
        metrics = PipelineMetrics()
        metrics.record_repair_attempt()
        metrics.record_repair_success()
        assert metrics.get_count(MetricEvent.REPAIR_ATTEMPT) == 1
        assert metrics.get_count(MetricEvent.REPAIR_SUCCESS) == 1

    def test_coverage_metrics(self):
        """Coverage check pass/failure metrics are tracked."""
        metrics = PipelineMetrics()
        metrics.record_coverage_check_pass()
        metrics.record_coverage_check_failure(gaps=["missing_ref"])
        assert metrics.get_count(MetricEvent.COVERAGE_CHECK_PASS) == 1
        assert metrics.get_count(MetricEvent.COVERAGE_CHECK_FAILURE) == 1

    def test_counters_increment(self):
        """Multiple emissions of the same event increment the counter."""
        metrics = PipelineMetrics()
        for _ in range(5):
            metrics.record_extraction_attempt()
        assert metrics.get_count(MetricEvent.EXTRACTION_ATTEMPT) == 5

    def test_reset_clears_counters(self):
        """reset() zeroes all counters."""
        metrics = PipelineMetrics()
        metrics.record_extraction_attempt()
        metrics.record_grounding_attempt()
        metrics.reset()
        assert metrics.get_count(MetricEvent.EXTRACTION_ATTEMPT) == 0
        assert metrics.get_count(MetricEvent.GROUNDING_ATTEMPT) == 0

    def test_unrecorded_event_returns_zero(self):
        """get_count for an event never emitted returns 0."""
        metrics = PipelineMetrics()
        assert metrics.get_count(MetricEvent.REPAIR_SUCCESS) == 0


# ---------------------------------------------------------------------------
# ShadowModeRecorder tests (Requirement 12.3)
# ---------------------------------------------------------------------------


class TestShadowModeRecorder:
    """Tests for shadow mode comparison metric recording."""

    def test_agree_when_both_true(self):
        """Agreement status is 'agree' when both results are True."""
        recorder = ShadowModeRecorder()
        metric = recorder.record_comparison(
            deterministic_result=True, llm_result=True
        )
        assert metric.agreement_status == "agree"

    def test_agree_when_both_false(self):
        """Agreement status is 'agree' when both results are False."""
        recorder = ShadowModeRecorder()
        metric = recorder.record_comparison(
            deterministic_result=False, llm_result=False
        )
        assert metric.agreement_status == "agree"

    def test_disagree_when_results_differ(self):
        """Agreement status is 'disagree' when results differ."""
        recorder = ShadowModeRecorder()
        metric = recorder.record_comparison(
            deterministic_result=True, llm_result=False
        )
        assert metric.agreement_status == "disagree"

    def test_llm_unavailable_when_none(self):
        """Agreement status is 'llm_unavailable' when llm_result is None."""
        recorder = ShadowModeRecorder()
        metric = recorder.record_comparison(
            deterministic_result=True, llm_result=None
        )
        assert metric.agreement_status == "llm_unavailable"

    def test_returns_shadow_comparison_metric(self):
        """record_comparison returns a ShadowComparisonMetric instance."""
        recorder = ShadowModeRecorder()
        metric = recorder.record_comparison(
            deterministic_result=False,
            llm_result=True,
            deterministic_gaps=["gap1"],
            llm_gaps=["gap2"],
        )
        assert isinstance(metric, ShadowComparisonMetric)
        assert metric.deterministic_gaps == ["gap1"]
        assert metric.llm_gaps == ["gap2"]

    def test_comparisons_list_accumulates(self):
        """All recorded comparisons are accessible via the comparisons property."""
        recorder = ShadowModeRecorder()
        recorder.record_comparison(deterministic_result=True, llm_result=True)
        recorder.record_comparison(deterministic_result=False, llm_result=True)
        assert len(recorder.comparisons) == 2

    def test_agreement_rate_calculation(self):
        """get_agreement_rate returns correct fraction."""
        recorder = ShadowModeRecorder()
        recorder.record_comparison(deterministic_result=True, llm_result=True)
        recorder.record_comparison(deterministic_result=True, llm_result=True)
        recorder.record_comparison(deterministic_result=True, llm_result=False)
        assert abs(recorder.get_agreement_rate() - 2.0 / 3.0) < 0.001

    def test_agreement_rate_empty(self):
        """get_agreement_rate returns 0.0 when no comparisons exist."""
        recorder = ShadowModeRecorder()
        assert recorder.get_agreement_rate() == 0.0


# ---------------------------------------------------------------------------
# DecisionOwnerRecorder tests (Requirement 12.4)
# ---------------------------------------------------------------------------


class TestDecisionOwnerRecorder:
    """Tests for decision-owner recording per semantic element."""

    def test_record_decision_stores_record(self):
        """record_decision creates and stores a DecisionOwnerRecord."""
        recorder = DecisionOwnerRecorder()
        rec = recorder.record_decision(
            element_path="actions[0].columns[0]",
            decision_owner="ColumnGrounder",
            resolution="amount",
            confidence=0.9,
            pipeline_stage="column_grounding",
        )
        assert isinstance(rec, DecisionOwnerRecord)
        assert rec.decision_owner == "ColumnGrounder"
        assert rec.element_path == "actions[0].columns[0]"
        assert rec.resolution == "amount"
        assert rec.confidence == 0.9
        assert rec.pipeline_stage == "column_grounding"

    def test_records_property_returns_copies(self):
        """records property returns a copy; mutations don't affect internal state."""
        recorder = DecisionOwnerRecorder()
        recorder.record_decision(
            element_path="a", decision_owner="X", resolution="r"
        )
        recs = recorder.records
        recs.clear()
        assert len(recorder.records) == 1

    def test_get_decisions_by_owner(self):
        """Filter decisions by owner name."""
        recorder = DecisionOwnerRecorder()
        recorder.record_decision(
            element_path="a", decision_owner="ColumnGrounder", resolution="col1"
        )
        recorder.record_decision(
            element_path="b", decision_owner="PredicateGrounder", resolution="col2"
        )
        recorder.record_decision(
            element_path="c", decision_owner="ColumnGrounder", resolution="col3"
        )
        by_cg = recorder.get_decisions_by_owner("ColumnGrounder")
        assert len(by_cg) == 2
        assert all(r.decision_owner == "ColumnGrounder" for r in by_cg)

    def test_get_decisions_for_element(self):
        """Filter decisions by element path."""
        recorder = DecisionOwnerRecorder()
        recorder.record_decision(
            element_path="actions[0].filter",
            decision_owner="PredicateGrounder",
            resolution="status",
        )
        recorder.record_decision(
            element_path="actions[1].columns[0]",
            decision_owner="ColumnGrounder",
            resolution="name",
        )
        results = recorder.get_decisions_for_element("actions[0].filter")
        assert len(results) == 1
        assert results[0].resolution == "status"

    def test_tracing_context_included_in_log(self):
        """When context is provided, it should be included in the log output."""
        recorder = DecisionOwnerRecorder()
        ctx = PipelineTracingContext(
            submission_id="sub-test", pipeline_stage="column_grounding"
        )
        rec = recorder.record_decision(
            element_path="x",
            decision_owner="Owner",
            resolution="res",
            context=ctx,
        )
        # Just verify it doesn't crash and record is created
        assert rec.decision_owner == "Owner"


# ---------------------------------------------------------------------------
# JSONFormatter tests
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Tests for structured JSON log formatting."""

    def test_output_is_valid_json(self):
        """Formatted log output should be parseable as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"

    def test_structured_data_merged(self):
        """Extra structured_data fields should appear in the JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="metric",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"metric_event": "extraction_attempt", "count": 5}  # type: ignore
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["metric_event"] == "extraction_attempt"
        assert parsed["count"] == 5


# ---------------------------------------------------------------------------
# PipelineStage enum tests
# ---------------------------------------------------------------------------


class TestPipelineStage:
    """Tests for pipeline stage enumeration."""

    def test_all_stages_defined(self):
        """All pipeline stages from the design should be present."""
        expected_stages = {
            "extraction",
            "coverage_validation",
            "semantic_repair",
            "resolution_coordination",
            "preflight_data_load",
            "schema_inference",
            "column_grounding",
            "predicate_grounding",
            "canonicalization",
            "compilation",
            "execution",
            "clarification",
        }
        actual_stages = {s.value for s in PipelineStage}
        assert expected_stages.issubset(actual_stages)
