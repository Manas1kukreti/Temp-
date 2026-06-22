"""
ClarificationRouter — API endpoints for interactive ambiguity resolution.

Exposes the POST /{submission_id}/clarify endpoint that accepts user answers
to clarification questions and delegates to ClarificationService for validation,
intent patching, and outcome routing.

Also provides GET /{submission_id}/clarification-status for frontend resync
after WebSocket disconnect (graceful degradation).

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.3, Error handling graceful degradation
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.security import require_roles
from app.db.session import get_db
from app.models import Submission, User, UserRole
from app.models.clarification import (
    ClarificationOutcome,
    ClarificationSession,
    SessionStatus,
)
from app.schemas.clarification import (
    ClarificationOutcomeResponse,
    ClarificationQuestionSchema,
    ClarificationResponsePayload,
)
from app.services.clarification_service import ClarificationService
from app.services.clarification_session_repo import SessionRepository

router = APIRouter(prefix="/uploads", tags=["clarification"])


@router.post(
    "/{submission_id}/clarify",
    response_model=ClarificationOutcomeResponse,
)
async def submit_clarification(
    submission_id: UUID,
    payload: ClarificationResponsePayload,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> ClarificationOutcomeResponse | JSONResponse:
    """Submit answers to clarification questions for a submission.

    Validates that the submission exists and has an active clarification session,
    checks session expiration, and delegates to ClarificationService.handle_response()
    for full validation, intent patching, and outcome routing.

    HTTP Status Code Mapping:
        - 200: RESOLVED, STILL_AMBIGUOUS, MAX_ROUNDS_EXCEEDED,
                CONFLICT_INTRODUCED, INVALID_RESPONSE (with error_details)
        - 400: Invalid question_id (does not belong to session)
        - 404: Submission not found
        - 409: Stale revision_token or intent_version mismatch
        - 410: Session expired
    """
    # --- Validate submission exists ---
    stmt = select(Submission).where(Submission.id == submission_id)
    result = await db.execute(stmt)
    submission = result.scalars().first()

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    # --- Validate submission has an active clarification session ---
    session_stmt = (
        select(ClarificationSession)
        .options(selectinload(ClarificationSession.questions))
        .where(
            ClarificationSession.submission_id == submission_id,
            ClarificationSession.status == SessionStatus.active,
        )
        .order_by(ClarificationSession.created_at.desc())
        .limit(1)
    )
    session_result = await db.execute(session_stmt)
    session = session_result.scalars().first()

    if session is None:
        raise HTTPException(
            status_code=410,
            detail="No active clarification session for this submission",
        )

    # --- Check session not expired (Req 10.3) ---
    now = datetime.now(UTC)
    expires_at = session.expires_at
    if expires_at is not None:
        # Ensure timezone-aware comparison
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if now > expires_at:
            # Return 410 Gone with SESSION_EXPIRED outcome
            response_data = ClarificationOutcomeResponse(
                outcome=ClarificationOutcome.SESSION_EXPIRED,
                submission_id=submission_id,
                session_id=session.id,
                intent_version=session.intent_version,
            )
            return JSONResponse(
                status_code=410,
                content=response_data.model_dump(mode="json"),
            )

    # --- Delegate to ClarificationService.handle_response() ---
    service = ClarificationService(db)
    outcome_response = await service.handle_response(submission_id, payload)

    # --- Map ClarificationOutcome to HTTP status codes ---
    outcome = outcome_response.outcome

    # Stale token/version → 409 Conflict
    if outcome == ClarificationOutcome.SESSION_EXPIRED:
        # Distinguish between actual expiration and token/version mismatch
        # If we got here (past the expiration check above), it's a token/version issue
        return JSONResponse(
            status_code=409,
            content=outcome_response.model_dump(mode="json"),
        )

    # Invalid question_id → 400 Bad Request
    if outcome == ClarificationOutcome.INVALID_RESPONSE and outcome_response.error_details:
        # Check if any error is about question not belonging to session
        for error in outcome_response.error_details:
            if "does not belong" in error.reason.lower():
                return JSONResponse(
                    status_code=400,
                    content=outcome_response.model_dump(mode="json"),
                )

    # INVALID_RESPONSE (validation errors) → 200 with error_details
    # RESOLVED, STILL_AMBIGUOUS, MAX_ROUNDS_EXCEEDED, CONFLICT_INTRODUCED → 200
    await db.commit()
    return outcome_response


@router.get(
    "/{submission_id}/clarification-status",
)
async def get_clarification_status(
    submission_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> JSONResponse:
    """Get the current clarification session state for a submission.

    Used by the frontend to resync after a WebSocket disconnect.
    Returns the active session's questions, round count, expiration, and status.

    HTTP Status Code Mapping:
        - 200: Active session found, returns full session state
        - 404: No active clarification session for this submission
    """
    repo = SessionRepository(db)
    session = await repo.get_active_by_submission(submission_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="No active clarification session",
        )

    # Serialize questions using the schema
    questions_serialized = [
        ClarificationQuestionSchema(
            id=q.id,
            session_id=q.session_id,
            round_number=q.round_number,
            intent_path=q.intent_path,
            reason_code=q.reason_code.value,
            question_text=q.question_text,
            candidate_options=q.candidate_options if isinstance(q.candidate_options, list) else [],
            free_text_enabled=q.free_text_enabled,
            user_answer=q.user_answer,
        ).model_dump(mode="json")
        for q in session.questions
    ]

    return JSONResponse(
        status_code=200,
        content={
            "session_id": str(session.id),
            "submission_id": str(session.submission_id),
            "intent_version": session.intent_version,
            "round_count": session.round_count,
            "max_rounds": session.max_rounds,
            "revision_token": session.revision_token,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "questions": questions_serialized,
            "status": session.status.value,
        },
    )
