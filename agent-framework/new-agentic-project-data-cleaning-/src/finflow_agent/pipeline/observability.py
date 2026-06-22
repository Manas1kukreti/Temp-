"""Observability module for FinFlow's semantic pipeline.

Provides structured tracing context, metric emitters, shadow mode comparison
recording, and decision-owner tracking for all pipeline stages.

Uses standard logging with JSON formatting for metrics emission rather than
requiring external metric systems.

Requirements: 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Literal

from finflow_agent.models.envelope import ShadowComparisonMetric


# ---------------------------------------------------------------------------
# JSON Logging Formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra structured fields attached to the record
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)
        return json.dumps(log_entry, default=str)


def get_pipeline_logger(name: str = "finflow.pipeline") -> logging.Logger:
    """Return a logger configured for structured JSON pipeline output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Pipeline Stage Enum
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Enumeration of pipeline stages for tracing context."""

    EXTRACTION = "extraction"
    COVERAGE_VALIDATION = "coverage_validation"
    SEMANTIC_REPAIR = "semantic_repair"
    RESOLUTION_COORDINATION = "resolution_coordination"
    PREFLIGHT_DATA_LOAD = "preflight_data_load"
    SCHEMA_INFERENCE = "schema_inference"
    COLUMN_GROUNDING = "column_grounding"
    PREDICATE_GROUNDING = "predicate_grounding"
    CANONICALIZATION = "canonicalization"
    COMPILATION = "compilation"
    EXECUTION = "execution"
    CLARIFICATION = "clarification"


# ---------------------------------------------------------------------------
# Structured Tracing Context (Requirement 12.2)
# ---------------------------------------------------------------------------


@dataclass
class PipelineTracingContext:
    """Structured tracing context attached to every pipeline log entry.

    Contains all required fields as specified in Requirement 12.2:
    submission_id, draft_id, draft_revision, intent_id, schema_fingerprint,
    profile_fingerprint, data_snapshot_ref, model_version, pipeline_stage,
    decision_owner, and duration_ms.
    """

    submission_id: str
    draft_id: str = ""
    draft_revision: int = 0
    intent_id: str = ""
    schema_fingerprint: str = ""
    profile_fingerprint: str = ""
    data_snapshot_ref: str = ""
    model_version: str = ""
    pipeline_stage: str = ""
    decision_owner: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracing context to a dictionary for log attachment."""
        return {
            "submission_id": self.submission_id,
            "draft_id": self.draft_id,
            "draft_revision": self.draft_revision,
            "intent_id": self.intent_id,
            "schema_fingerprint": self.schema_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "data_snapshot_ref": self.data_snapshot_ref,
            "model_version": self.model_version,
            "pipeline_stage": self.pipeline_stage,
            "decision_owner": self.decision_owner,
            "duration_ms": self.duration_ms,
        }

    @contextmanager
    def timed_stage(
        self, stage: str, decision_owner: str = ""
    ) -> Generator[PipelineTracingContext, None, None]:
        """Context manager that measures duration for a pipeline stage.

        Sets pipeline_stage and decision_owner, then records duration_ms
        on exit.
        """
        self.pipeline_stage = stage
        self.decision_owner = decision_owner
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.duration_ms = elapsed_ms


# ---------------------------------------------------------------------------
# Pipeline Metrics (Requirement 12.1)
# ---------------------------------------------------------------------------


class MetricEvent(str, Enum):
    """All metric event types emitted by the pipeline."""

    # Extraction metrics
    EXTRACTION_ATTEMPT = "extraction_attempt"
    EXTRACTION_SUCCESS = "extraction_success"
    EXTRACTION_FAILURE = "extraction_failure"

    # Grounding metrics
    GROUNDING_ATTEMPT = "grounding_attempt"
    GROUNDING_SUCCESS = "grounding_success"
    GROUNDING_LLM_FALLBACK_INVOCATION = "grounding_llm_fallback_invocation"

    # Post-LLM verification metrics
    POST_LLM_VERIFICATION_PASS = "post_llm_verification_pass"
    POST_LLM_VERIFICATION_FAILURE = "post_llm_verification_failure"

    # Clarification metrics
    CLARIFICATION_SESSION_INITIATED = "clarification_session_initiated"
    CLARIFICATION_SESSION_RESOLVED = "clarification_session_resolved"

    # Repair metrics
    REPAIR_ATTEMPT = "repair_attempt"
    REPAIR_SUCCESS = "repair_success"

    # Coverage metrics
    COVERAGE_CHECK_PASS = "coverage_check_pass"
    COVERAGE_CHECK_FAILURE = "coverage_check_failure"


@dataclass
class PipelineMetrics:
    """Records and emits structured metrics for all pipeline events.

    Emits metrics via JSON-structured logging. Maintains in-memory counters
    for the current pipeline invocation for quick access.

    Requirement 12.1: metrics for extraction, grounding, repair, coverage,
    and clarification events.
    """

    _logger: logging.Logger = field(init=False)
    _counters: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._logger = get_pipeline_logger("finflow.pipeline.metrics")

    def emit(
        self,
        event: MetricEvent,
        context: PipelineTracingContext | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Emit a metric event with optional tracing context and extra data.

        Increments the in-memory counter for the event and logs the metric
        as structured JSON.
        """
        event_name = event.value
        self._counters[event_name] = self._counters.get(event_name, 0) + 1

        log_data: dict[str, Any] = {
            "metric_event": event_name,
            "metric_count": self._counters[event_name],
        }
        if context is not None:
            log_data["tracing"] = context.to_dict()
        if extra:
            log_data["extra"] = extra

        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"metric:{event_name}",
            args=(),
            exc_info=None,
        )
        record.structured_data = log_data  # type: ignore[attr-defined]
        self._logger.handle(record)

    def get_count(self, event: MetricEvent) -> int:
        """Return the current count for a given metric event."""
        return self._counters.get(event.value, 0)

    def reset(self) -> None:
        """Reset all in-memory counters."""
        self._counters.clear()

    # -----------------------------------------------------------------------
    # Convenience emitters for each pipeline event category
    # -----------------------------------------------------------------------

    def record_extraction_attempt(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.EXTRACTION_ATTEMPT, context)

    def record_extraction_success(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.EXTRACTION_SUCCESS, context)

    def record_extraction_failure(
        self,
        context: PipelineTracingContext | None = None,
        reason: str = "",
    ) -> None:
        self.emit(
            MetricEvent.EXTRACTION_FAILURE,
            context,
            extra={"failure_reason": reason} if reason else None,
        )

    def record_grounding_attempt(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.GROUNDING_ATTEMPT, context)

    def record_grounding_success(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.GROUNDING_SUCCESS, context)

    def record_grounding_llm_fallback(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.GROUNDING_LLM_FALLBACK_INVOCATION, context)

    def record_post_llm_verification_pass(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.POST_LLM_VERIFICATION_PASS, context)

    def record_post_llm_verification_failure(
        self,
        context: PipelineTracingContext | None = None,
        checks_failed: list[str] | None = None,
    ) -> None:
        self.emit(
            MetricEvent.POST_LLM_VERIFICATION_FAILURE,
            context,
            extra={"checks_failed": checks_failed} if checks_failed else None,
        )

    def record_clarification_initiated(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.CLARIFICATION_SESSION_INITIATED, context)

    def record_clarification_resolved(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.CLARIFICATION_SESSION_RESOLVED, context)

    def record_repair_attempt(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.REPAIR_ATTEMPT, context)

    def record_repair_success(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.REPAIR_SUCCESS, context)

    def record_coverage_check_pass(
        self, context: PipelineTracingContext | None = None
    ) -> None:
        self.emit(MetricEvent.COVERAGE_CHECK_PASS, context)

    def record_coverage_check_failure(
        self,
        context: PipelineTracingContext | None = None,
        gaps: list[str] | None = None,
    ) -> None:
        self.emit(
            MetricEvent.COVERAGE_CHECK_FAILURE,
            context,
            extra={"gaps": gaps} if gaps else None,
        )


# ---------------------------------------------------------------------------
# Shadow Mode Comparison Recording (Requirement 12.3)
# ---------------------------------------------------------------------------


@dataclass
class ShadowModeRecorder:
    """Records comparison metrics between deterministic and LLM coverage
    results when shadow mode is active.

    Emits ShadowComparisonMetric from the envelope module as structured
    log entries for offline analysis.

    Requirement 12.3: emit comparison metric recording deterministic_result,
    llm_result, and agreement_status.
    """

    _logger: logging.Logger = field(init=False)
    _comparisons: list[ShadowComparisonMetric] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        self._logger = get_pipeline_logger("finflow.pipeline.shadow")

    def record_comparison(
        self,
        deterministic_result: bool,
        llm_result: bool | None,
        deterministic_gaps: list[str] | None = None,
        llm_gaps: list[str] | None = None,
        context: PipelineTracingContext | None = None,
    ) -> ShadowComparisonMetric:
        """Record a shadow mode comparison between deterministic and LLM results.

        Determines agreement status and emits the metric via structured logging.
        Returns the created ShadowComparisonMetric.
        """
        if llm_result is None:
            agreement_status: Literal["agree", "disagree", "llm_unavailable"] = (
                "llm_unavailable"
            )
        elif deterministic_result == llm_result:
            agreement_status = "agree"
        else:
            agreement_status = "disagree"

        metric = ShadowComparisonMetric(
            deterministic_result=deterministic_result,
            llm_result=llm_result,
            agreement_status=agreement_status,
            deterministic_gaps=deterministic_gaps or [],
            llm_gaps=llm_gaps or [],
        )

        self._comparisons.append(metric)

        log_data: dict[str, Any] = {
            "shadow_comparison": metric.model_dump(mode="json"),
        }
        if context is not None:
            log_data["tracing"] = context.to_dict()

        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"shadow_comparison:{agreement_status}",
            args=(),
            exc_info=None,
        )
        record.structured_data = log_data  # type: ignore[attr-defined]
        self._logger.handle(record)

        return metric

    @property
    def comparisons(self) -> list[ShadowComparisonMetric]:
        """Return all recorded shadow comparisons."""
        return list(self._comparisons)

    def get_agreement_rate(self) -> float:
        """Return the fraction of comparisons where deterministic and LLM agree.

        Returns 0.0 if no comparisons have been recorded.
        """
        if not self._comparisons:
            return 0.0
        agrees = sum(1 for c in self._comparisons if c.agreement_status == "agree")
        return agrees / len(self._comparisons)


# ---------------------------------------------------------------------------
# Decision Owner Recording (Requirement 12.4)
# ---------------------------------------------------------------------------


@dataclass
class DecisionOwnerRecord:
    """Single record of a decision-owner assignment for a semantic element."""

    element_path: str
    decision_owner: str
    resolution: str
    confidence: float
    pipeline_stage: str


@dataclass
class DecisionOwnerRecorder:
    """Records the Decision_Owner for each finalized semantic element.

    Requirement 12.4: record decision_owner in the structured trace for each
    finalized semantic element.
    """

    _logger: logging.Logger = field(init=False)
    _records: list[DecisionOwnerRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._logger = get_pipeline_logger("finflow.pipeline.decisions")

    def record_decision(
        self,
        element_path: str,
        decision_owner: str,
        resolution: str,
        confidence: float = 1.0,
        pipeline_stage: str = "",
        context: PipelineTracingContext | None = None,
    ) -> DecisionOwnerRecord:
        """Record a decision-owner assignment for a finalized semantic element.

        Logs the decision as structured JSON and stores it for later retrieval.
        """
        record = DecisionOwnerRecord(
            element_path=element_path,
            decision_owner=decision_owner,
            resolution=resolution,
            confidence=confidence,
            pipeline_stage=pipeline_stage,
        )
        self._records.append(record)

        log_data: dict[str, Any] = {
            "decision_owner_record": {
                "element_path": element_path,
                "decision_owner": decision_owner,
                "resolution": resolution,
                "confidence": confidence,
                "pipeline_stage": pipeline_stage,
            },
        }
        if context is not None:
            log_data["tracing"] = context.to_dict()

        log_record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"decision_owner:{decision_owner}:{element_path}",
            args=(),
            exc_info=None,
        )
        log_record.structured_data = log_data  # type: ignore[attr-defined]
        self._logger.handle(log_record)

        return record

    @property
    def records(self) -> list[DecisionOwnerRecord]:
        """Return all recorded decision-owner assignments."""
        return list(self._records)

    def get_decisions_by_owner(self, owner: str) -> list[DecisionOwnerRecord]:
        """Return all decisions made by a specific owner."""
        return [r for r in self._records if r.decision_owner == owner]

    def get_decisions_for_element(self, element_path: str) -> list[DecisionOwnerRecord]:
        """Return all decisions recorded for a specific element path."""
        return [r for r in self._records if r.element_path == element_path]
