"""
ClarificationService — central orchestrator for interactive ambiguity resolution.

Coordinates session creation, question generation, response handling,
intent patching, and outcome routing for the clarification subsystem.

Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 5.3, 6.5, 7.1–7.5, 10.1–10.3
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Submission, SubmissionStatus
from app.models.clarification import (
    ClarificationOutcome,
    ClarificationQuestion,
    ClarificationSession,
    SessionStatus,
)
from app.schemas.clarification import (
    ClarificationOutcomeResponse,
    ClarificationQuestionSchema,
    ClarificationResponsePayload,
    QuestionError,
)
from app.services.clarification_questions import QuestionGenerator, UnresolvedField
from app.services.clarification_session_repo import SessionRepository, generate_revision_token
from app.services.clarification_validator import ResponseValidator, ValidationResult
from app.services.intent_patcher import IntentPatcher
from app.services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CLARIFICATION_THRESHOLD: float = 0.5
"""Ambiguity score threshold. Scores >= this trigger clarification; below → quarantine."""

SESSION_TIMEOUT_MINUTES: int = 30
"""Default session expiration timeout in minutes."""


class ClarificationService:
    """Coordinates session creation, question generation, response handling, and outcome routing."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.session_repo = SessionRepository(db)
        self.question_generator = QuestionGenerator()

    async def initiate_session(
        self,
        submission_id: UUID,
        intent: Any,
        intent_package: Any,
        ambiguity_score: float,
    ) -> ClarificationSession | None:
        """Create a clarification session, generate questions, and broadcast via WebSocket.

        If the ambiguity_score is below the threshold, the submission is quarantined
        directly and None is returned.

        Args:
            submission_id: The UUID of the submission needing clarification.
            intent: The CanonicalIntent (dict or object) with unresolved fields.
            intent_package: The IntentPackage schema-resolution artifact.
            ambiguity_score: Numeric score indicating how resolvable the ambiguity is.

        Returns:
            The created ClarificationSession if score >= threshold, else None.
        """
        # --- Check ambiguity score against threshold (Req 1.1, 1.2) ---
        if ambiguity_score < CLARIFICATION_THRESHOLD:
            await self._quarantine_submission(
                submission_id, reason="ambiguity_score_below_threshold"
            )
            return None

        # --- Determine intent version and unresolved fields ---
        intent_version = _get_intent_version(intent)
        unresolved_fields = _extract_unresolved_fields(intent, intent_package)

        # --- Generate revision token ---
        revision_token = generate_revision_token()

        # --- Compute expiration time (Req 10.1) ---
        expires_at = datetime.now(UTC) + timedelta(minutes=SESSION_TIMEOUT_MINUTES)

        # --- Create ClarificationSession (Req 1.4) ---
        session = ClarificationSession(
            submission_id=submission_id,
            intent_version=intent_version,
            round_count=0,
            max_rounds=2,
            status=SessionStatus.active,
            revision_token=revision_token,
            expires_at=expires_at,
        )
        session = await self.session_repo.create(session)

        # --- Generate questions for unresolved fields (Req 2.1) ---
        generated_questions = self.question_generator.generate(
            unresolved_fields, intent_package, intent
        )

        # --- Persist questions associated with the session ---
        db_questions: list[ClarificationQuestion] = []
        for gq in generated_questions:
            question = ClarificationQuestion(
                session_id=session.id,
                round_number=0,
                intent_path=gq.intent_path,
                reason_code=gq.reason_code,
                question_text=gq.question_text,
                candidate_options=gq.candidate_options,
                free_text_enabled=gq.free_text_enabled,
            )
            self.db.add(question)
            db_questions.append(question)

        await self.db.flush()
        # Refresh to get generated IDs
        for q in db_questions:
            await self.db.refresh(q)

        # --- Transition submission status to "awaiting_clarification" (Req 1.3) ---
        await self._transition_submission_status(
            submission_id, SubmissionStatus.awaiting_clarification
        )

        # --- Broadcast "clarification_required" WebSocket event (Req 3.1, 3.2) ---
        await self._broadcast_clarification_required(session, db_questions)

        return session

    async def handle_response(
        self,
        submission_id: UUID,
        response: ClarificationResponsePayload,
    ) -> ClarificationOutcomeResponse:
        """Validate, patch intent, re-ground, and determine outcome.

        Implements the full response-handling workflow:
        1. Load active session
        2. Validate via ResponseValidator
        3. On valid: increment round, apply patch, route outcome
        4. Record round history
        5. Broadcast WebSocket event

        Requirements: 5.3, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 8.4, 12.2, 12.3
        """
        # --- Step 1: Load active session ---
        session = await self.session_repo.get_active_by_submission(submission_id)
        if session is None:
            return ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.SESSION_EXPIRED,
                submission_id=submission_id,
                session_id=response.session_id,
                intent_version=response.intent_version,
                error_details=[QuestionError(question_id=response.session_id, reason="No active session found")],
            )

        # --- Step 2: Validate via ResponseValidator ---
        validator = ResponseValidator()
        validation_result: ValidationResult = validator.validate(session, response)

        if not validation_result.is_valid:
            # Token mismatch → SESSION_EXPIRED
            if validation_result.rejection_type == "token_mismatch":
                return ClarificationOutcomeResponse(
                    outcome=ClarificationOutcome.SESSION_EXPIRED,
                    submission_id=submission_id,
                    session_id=session.id,
                    intent_version=session.intent_version,
                    error_details=[
                        QuestionError(
                            question_id=response.session_id,
                            reason="Revision token mismatch — session state has changed.",
                        )
                    ],
                )

            # Version mismatch → stale-version error
            if validation_result.rejection_type == "version_mismatch":
                return ClarificationOutcomeResponse(
                    outcome=ClarificationOutcome.SESSION_EXPIRED,
                    submission_id=submission_id,
                    session_id=session.id,
                    intent_version=session.intent_version,
                    error_details=[
                        QuestionError(
                            question_id=response.session_id,
                            reason="Intent version mismatch — stale version submitted.",
                        )
                    ],
                )

            # Invalid question_id → error (400-level in the router)
            if validation_result.rejection_type == "invalid_question_id":
                error_details = [
                    QuestionError(question_id=UUID(e["question_id"]), reason=e["reason"])
                    for e in (validation_result.error_details or [])
                ]
                return ClarificationOutcomeResponse(
                    outcome=ClarificationOutcome.INVALID_RESPONSE,
                    submission_id=submission_id,
                    session_id=session.id,
                    intent_version=session.intent_version,
                    error_details=error_details,
                )

            # Invalid answers → INVALID_RESPONSE (no round increment)
            if validation_result.rejection_type == "invalid_answers":
                error_details = [
                    QuestionError(question_id=UUID(e["question_id"]), reason=e["reason"])
                    for e in (validation_result.error_details or [])
                ]
                # Record round history with INVALID_RESPONSE outcome
                await self.session_repo.record_round(
                    session_id=session.id,
                    questions=list(session.questions),
                    answers=[
                        {
                            "question_id": str(a.question_id),
                            "selected_option": a.selected_option,
                            "free_text": a.free_text,
                        }
                        for a in response.answers
                    ],
                    outcome=ClarificationOutcome.INVALID_RESPONSE.value,
                )
                return ClarificationOutcomeResponse(
                    outcome=ClarificationOutcome.INVALID_RESPONSE,
                    submission_id=submission_id,
                    session_id=session.id,
                    intent_version=session.intent_version,
                    error_details=error_details,
                )

        # --- Step 3: Valid response — increment round_count and apply patch ---
        session.round_count += 1
        await self.db.flush()

        # Load submission to get current intent
        from sqlalchemy import select as sa_select

        stmt = sa_select(Submission).where(Submission.id == submission_id)
        result = await self.db.execute(stmt)
        submission = result.scalars().first()

        if not submission or not submission.canonical_intent:
            return ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.SESSION_EXPIRED,
                submission_id=submission_id,
                session_id=session.id,
                intent_version=session.intent_version,
                error_details=[
                    QuestionError(
                        question_id=response.session_id,
                        reason="Submission or intent not found.",
                    )
                ],
            )

        intent = submission.canonical_intent
        # Use a simple object as intent_package if submission doesn't have one
        intent_package = getattr(submission, "intent_package", None)

        # Call IntentPatcher.apply_patch()
        patcher = IntentPatcher()
        patch_result = await patcher.apply_patch(
            intent=intent,
            intent_package=intent_package,
            answers=response.answers,
            questions=list(session.questions),
            db=self.db,
            submission=submission,
        )

        # --- Step 4: Determine outcome based on PatchResult ---

        # 4a: Conflict introduced → CONFLICT_INTRODUCED (decrement round back)
        if patch_result.conflict_flag:
            session.round_count -= 1
            session.outcome = ClarificationOutcome.CONFLICT_INTRODUCED
            await self.db.flush()

            # Record round history
            await self.session_repo.record_round(
                session_id=session.id,
                questions=list(session.questions),
                answers=[
                    {
                        "question_id": str(a.question_id),
                        "selected_option": a.selected_option,
                        "free_text": a.free_text,
                    }
                    for a in response.answers
                ],
                outcome=ClarificationOutcome.CONFLICT_INTRODUCED.value,
            )

            return ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.CONFLICT_INTRODUCED,
                submission_id=submission_id,
                session_id=session.id,
                intent_version=patch_result.new_revision,
            )

        # Update submission's canonical_intent with patched version
        submission.canonical_intent = patch_result.intent
        await self.db.flush()

        # 4b: No remaining unresolved fields → RESOLVED
        if not patch_result.remaining_unresolved_fields:
            session.status = SessionStatus.resolved
            session.outcome = ClarificationOutcome.RESOLVED
            session.intent_version = patch_result.new_revision
            await self.db.flush()

            # Transition submission to "running" (dispatch to execution)
            await self._transition_submission_status(submission_id, SubmissionStatus.running)

            # Record round history
            await self.session_repo.record_round(
                session_id=session.id,
                questions=list(session.questions),
                answers=[
                    {
                        "question_id": str(a.question_id),
                        "selected_option": a.selected_option,
                        "free_text": a.free_text,
                    }
                    for a in response.answers
                ],
                outcome=ClarificationOutcome.RESOLVED.value,
            )

            # Broadcast "clarification_resolved" (Req 12.2 — only this event, no round_update)
            await self._broadcast_clarification_resolved(session, patch_result.new_revision)

            return ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.RESOLVED,
                submission_id=submission_id,
                session_id=session.id,
                intent_version=patch_result.new_revision,
            )

        # 4c: Remaining fields + round_count < max_rounds → STILL_AMBIGUOUS
        if session.round_count < session.max_rounds:
            # Generate new questions for remaining unresolved fields
            new_generated_questions = self.question_generator.generate(
                patch_result.remaining_unresolved_fields, intent_package, patch_result.intent
            )

            # Generate new revision token
            new_token = generate_revision_token()
            session.revision_token = new_token
            session.intent_version = patch_result.new_revision
            session.outcome = ClarificationOutcome.STILL_AMBIGUOUS
            await self.db.flush()

            # Persist new questions
            new_db_questions: list[ClarificationQuestion] = []
            for gq in new_generated_questions:
                question = ClarificationQuestion(
                    session_id=session.id,
                    round_number=session.round_count,
                    intent_path=gq.intent_path,
                    reason_code=gq.reason_code,
                    question_text=gq.question_text,
                    candidate_options=gq.candidate_options,
                    free_text_enabled=gq.free_text_enabled,
                )
                self.db.add(question)
                new_db_questions.append(question)

            await self.db.flush()
            for q in new_db_questions:
                await self.db.refresh(q)

            # Record round history
            await self.session_repo.record_round(
                session_id=session.id,
                questions=list(session.questions),
                answers=[
                    {
                        "question_id": str(a.question_id),
                        "selected_option": a.selected_option,
                        "free_text": a.free_text,
                    }
                    for a in response.answers
                ],
                outcome=ClarificationOutcome.STILL_AMBIGUOUS.value,
            )

            # Broadcast "clarification_round_update" (Req 12.3)
            await self._broadcast_clarification_round_update(
                session, new_db_questions, new_token
            )

            # Build remaining_questions for response
            remaining_questions = [
                ClarificationQuestionSchema(
                    id=q.id,
                    session_id=q.session_id,
                    round_number=q.round_number,
                    intent_path=q.intent_path,
                    reason_code=q.reason_code,
                    question_text=q.question_text,
                    candidate_options=q.candidate_options,
                    free_text_enabled=q.free_text_enabled,
                )
                for q in new_db_questions
            ]

            return ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.STILL_AMBIGUOUS,
                submission_id=submission_id,
                session_id=session.id,
                intent_version=patch_result.new_revision,
                remaining_questions=remaining_questions,
            )

        # 4d: Remaining fields + round_count == max_rounds → MAX_ROUNDS_EXCEEDED
        session.status = SessionStatus.max_rounds_exceeded
        session.outcome = ClarificationOutcome.MAX_ROUNDS_EXCEEDED
        session.intent_version = patch_result.new_revision
        await self.db.flush()

        # Quarantine submission
        await self._quarantine_submission(submission_id, reason="max_clarification_rounds_exceeded")

        # Record round history
        await self.session_repo.record_round(
            session_id=session.id,
            questions=list(session.questions),
            answers=[
                {
                    "question_id": str(a.question_id),
                    "selected_option": a.selected_option,
                    "free_text": a.free_text,
                }
                for a in response.answers
            ],
            outcome=ClarificationOutcome.MAX_ROUNDS_EXCEEDED.value,
        )

        # Broadcast "clarification_expired"
        await self._broadcast_clarification_expired(session)

        return ClarificationOutcomeResponse(
            outcome=ClarificationOutcome.MAX_ROUNDS_EXCEEDED,
            submission_id=submission_id,
            session_id=session.id,
            intent_version=patch_result.new_revision,
        )

    async def expire_session(self, session_id: UUID) -> None:
        """Transition session to expired, quarantine submission.

        Loads the session by ID, sets its status to expired with SESSION_EXPIRED
        outcome, quarantines the associated submission, and broadcasts the
        "clarification_expired" WebSocket event.

        Requirements: 10.2, 10.3
        """
        from sqlalchemy import select

        # --- Load session by ID ---
        stmt = select(ClarificationSession).where(ClarificationSession.id == session_id)
        result = await self.db.execute(stmt)
        session = result.scalars().first()

        if session is None:
            logger.warning("expire_session called for non-existent session: %s", session_id)
            return

        # --- Transition session status to "expired" and set outcome (Req 10.2) ---
        session.status = SessionStatus.expired
        session.outcome = ClarificationOutcome.SESSION_EXPIRED
        await self.db.flush()

        # --- Quarantine submission with reason "clarification_session_expired" (Req 10.2) ---
        await self._quarantine_submission(
            session.submission_id, reason="clarification_session_expired"
        )

        # --- Broadcast "clarification_expired" WebSocket event (Req 10.3) ---
        await self._broadcast_clarification_expired(session)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    async def _quarantine_submission(self, submission_id: UUID, reason: str) -> None:
        """Set submission status to quarantined with the given reason."""
        from sqlalchemy import select

        stmt = select(Submission).where(Submission.id == submission_id)
        result = await self.db.execute(stmt)
        submission = result.scalars().first()
        if submission:
            submission.status = SubmissionStatus.quarantined
            if hasattr(submission, "summary") and submission.summary is None:
                submission.summary = {}
            if hasattr(submission, "summary") and isinstance(submission.summary, dict):
                submission.summary = {**submission.summary, "quarantine_reason": reason}
            await self.db.flush()

    async def _transition_submission_status(
        self, submission_id: UUID, new_status: SubmissionStatus
    ) -> None:
        """Transition a submission to the given status."""
        from sqlalchemy import select

        stmt = select(Submission).where(Submission.id == submission_id)
        result = await self.db.execute(stmt)
        submission = result.scalars().first()
        if submission:
            submission.status = new_status
            await self.db.flush()

    async def _broadcast_clarification_required(
        self,
        session: ClarificationSession,
        questions: list[ClarificationQuestion],
    ) -> None:
        """Broadcast the 'clarification_required' event via WebSocket (Req 3.1, 3.2)."""
        payload = {
            "session_id": str(session.id),
            "submission_id": str(session.submission_id),
            "intent_version": session.intent_version,
            "revision_token": session.revision_token,
            "expires_at": session.expires_at.isoformat(),
            "questions": [
                {
                    "question_id": str(q.id),
                    "intent_path": q.intent_path,
                    "reason_code": q.reason_code.value if hasattr(q.reason_code, "value") else q.reason_code,
                    "question_text": q.question_text,
                    "candidate_options": q.candidate_options,
                    "free_text_enabled": q.free_text_enabled,
                }
                for q in questions
            ],
        }
        try:
            await ws_manager.broadcast("uploads", "clarification_required", payload)
        except Exception as e:
            # WebSocket broadcast failure is non-blocking (design spec)
            logger.warning("Failed to broadcast clarification_required event: %s", e)

    async def _broadcast_clarification_resolved(
        self,
        session: ClarificationSession,
        intent_version: int,
    ) -> None:
        """Broadcast the 'clarification_resolved' event via WebSocket (Req 12.2)."""
        payload = {
            "session_id": str(session.id),
            "submission_id": str(session.submission_id),
            "intent_version": intent_version,
            "outcome": ClarificationOutcome.RESOLVED.value,
        }
        try:
            await ws_manager.broadcast("uploads", "clarification_resolved", payload)
        except Exception as e:
            logger.warning("Failed to broadcast clarification_resolved event: %s", e)

    async def _broadcast_clarification_round_update(
        self,
        session: ClarificationSession,
        questions: list[ClarificationQuestion],
        revision_token: str,
    ) -> None:
        """Broadcast the 'clarification_round_update' event via WebSocket (Req 12.3)."""
        payload = {
            "session_id": str(session.id),
            "submission_id": str(session.submission_id),
            "round_count": session.round_count,
            "revision_token": revision_token,
            "questions": [
                {
                    "question_id": str(q.id),
                    "intent_path": q.intent_path,
                    "reason_code": q.reason_code.value if hasattr(q.reason_code, "value") else q.reason_code,
                    "question_text": q.question_text,
                    "candidate_options": q.candidate_options,
                    "free_text_enabled": q.free_text_enabled,
                }
                for q in questions
            ],
        }
        try:
            await ws_manager.broadcast("uploads", "clarification_round_update", payload)
        except Exception as e:
            logger.warning("Failed to broadcast clarification_round_update event: %s", e)

    async def _broadcast_clarification_expired(
        self,
        session: ClarificationSession,
    ) -> None:
        """Broadcast the 'clarification_expired' event via WebSocket (Req 3.3)."""
        payload = {
            "session_id": str(session.id),
            "submission_id": str(session.submission_id),
        }
        try:
            await ws_manager.broadcast("uploads", "clarification_expired", payload)
        except Exception as e:
            logger.warning("Failed to broadcast clarification_expired event: %s", e)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_intent_version(intent: Any) -> int:
    """Extract the intent_revision from the intent object or dict."""
    if isinstance(intent, dict):
        return intent.get("intent_revision", 1)
    if hasattr(intent, "intent_revision"):
        return intent.intent_revision or 1
    return 1


def _extract_unresolved_fields(intent: Any, intent_package: Any) -> list[UnresolvedField]:
    """Extract unresolved fields from the intent for question generation.

    Reuses the re-grounding logic from IntentPatcher to identify remaining
    unresolved column references.
    """
    from app.services.intent_patcher import _re_ground_and_validate

    if isinstance(intent, dict):
        return _re_ground_and_validate(intent, intent_package)

    # If intent is not a dict, attempt to get unresolved fields from intent_package
    if hasattr(intent_package, "unresolved_fields"):
        fields = intent_package.unresolved_fields
        if isinstance(fields, list):
            return [
                UnresolvedField(
                    intent_path=f.intent_path if hasattr(f, "intent_path") else str(f),
                    reason_code=f.reason_code if hasattr(f, "reason_code") else "LOW_CONFIDENCE_SCORE",
                    raw_reference=f.raw_reference if hasattr(f, "raw_reference") else "",
                    grounding_candidates=f.grounding_candidates if hasattr(f, "grounding_candidates") else [],
                )
                for f in fields
            ]

    return []
