"""
Repository for ClarificationSession persistence operations.

Provides CRUD and query methods for clarification sessions, including
session creation with cryptographically random revision tokens,
active session lookups, expiration queries, and round history recording.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.clarification import (
    ClarificationQuestion,
    ClarificationSession,
    SessionStatus,
)


def generate_revision_token() -> str:
    """Generate a cryptographically random revision token (at least 32 characters).

    Uses secrets.token_urlsafe which produces URL-safe base64-encoded tokens.
    With nbytes=32, the output is approximately 43 characters long.
    """
    return secrets.token_urlsafe(32)


class SessionRepository:
    """Database operations for clarification sessions."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, session: ClarificationSession) -> ClarificationSession:
        """Persist a new ClarificationSession.

        If the session does not already have a revision_token set,
        a cryptographically random token is generated automatically.
        """
        if not session.revision_token:
            session.revision_token = generate_revision_token()
        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)
        return session

    async def get_active_by_submission(
        self, submission_id: UUID
    ) -> ClarificationSession | None:
        """Fetch the active clarification session for a given submission.

        Returns None if no active session exists.
        """
        stmt = (
            select(ClarificationSession)
            .options(selectinload(ClarificationSession.questions))
            .where(
                ClarificationSession.submission_id == submission_id,
                ClarificationSession.status == SessionStatus.active,
            )
            .order_by(ClarificationSession.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def update(self, session: ClarificationSession) -> None:
        """Update an existing session's state in the database.

        Merges the session object back into the current database session
        and flushes changes.
        """
        merged = await self.db.merge(session)
        await self.db.flush()
        await self.db.refresh(merged)

    async def get_expired_sessions(self) -> list[ClarificationSession]:
        """Query all sessions that have exceeded their expires_at with status=active.

        Used by the ExpirationScheduler to find sessions that need cleanup.
        """
        now = datetime.now(UTC)
        stmt = (
            select(ClarificationSession)
            .options(selectinload(ClarificationSession.questions))
            .where(
                ClarificationSession.status == SessionStatus.active,
                ClarificationSession.expires_at < now,
            )
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def record_round(
        self,
        session_id: UUID,
        questions: list[ClarificationQuestion],
        answers: list[dict[str, Any]],
        outcome: str,
    ) -> None:
        """Persist question-answer history for a completed clarification round.

        Updates existing questions with user answers and records the outcome
        for auditability (Requirement 14.4).

        Args:
            session_id: The session this round belongs to.
            questions: The ClarificationQuestion objects for this round.
            answers: A list of answer dicts, each containing question_id,
                     selected_option, and/or free_text.
            outcome: The ClarificationOutcome value for this round.
        """
        # Index answers by question_id for efficient lookup
        answers_by_question_id = {
            str(answer["question_id"]): answer for answer in answers
        }

        # Update each question with its corresponding user answer
        for question in questions:
            answer = answers_by_question_id.get(str(question.id))
            if answer:
                question.user_answer = {
                    "selected_option": answer.get("selected_option"),
                    "free_text": answer.get("free_text"),
                    "outcome": outcome,
                    "answered_at": datetime.now(UTC).isoformat(),
                }
                merged = await self.db.merge(question)

        await self.db.flush()
