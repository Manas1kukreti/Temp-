"""
Clarification subsystem data models.

Defines the ClarificationSession, ClarificationQuestion, and supporting enums
for the interactive ambiguity resolution workflow.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models import Base, enum_values


class ClarificationOutcome(str, enum.Enum):
    """Final outcome of a clarification session or individual round."""

    RESOLVED = "RESOLVED"
    STILL_AMBIGUOUS = "STILL_AMBIGUOUS"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    CONFLICT_INTRODUCED = "CONFLICT_INTRODUCED"
    MAX_ROUNDS_EXCEEDED = "MAX_ROUNDS_EXCEEDED"
    SESSION_EXPIRED = "SESSION_EXPIRED"


class ReasonCode(str, enum.Enum):
    """Classification of why a field is unresolved."""

    MULTIPLE_COLUMN_MATCHES = "MULTIPLE_COLUMN_MATCHES"
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"
    LOW_CONFIDENCE_SCORE = "LOW_CONFIDENCE_SCORE"
    MISSING_COLUMN = "MISSING_COLUMN"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class SessionStatus(str, enum.Enum):
    """Current lifecycle status of a clarification session."""

    active = "active"
    resolved = "resolved"
    expired = "expired"
    max_rounds_exceeded = "max_rounds_exceeded"


class ClarificationSession(Base):
    """
    Tracks an interactive clarification exchange between the system and the user
    for a specific submission with resolvable ambiguities.
    """

    __tablename__ = "clarification_sessions"
    __table_args__ = (
        Index("ix_clarification_sessions_submission_id", "submission_id"),
        Index("ix_clarification_sessions_status", "status"),
        Index("ix_clarification_sessions_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
    )
    intent_version: Mapped[int] = mapped_column(Integer, nullable=False)
    round_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status", values_callable=enum_values),
        default=SessionStatus.active,
        nullable=False,
    )
    revision_token: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    outcome: Mapped[ClarificationOutcome | None] = mapped_column(
        Enum(ClarificationOutcome, name="clarification_outcome", values_callable=enum_values),
        nullable=True,
    )

    # Relationships
    questions: Mapped[list["ClarificationQuestion"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ClarificationQuestion(Base):
    """
    A structured question targeting a specific unresolved intent path,
    containing candidate options and a free-text input field.
    """

    __tablename__ = "clarification_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clarification_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    intent_path: Mapped[str] = mapped_column(Text, nullable=False)
    reason_code: Mapped[ReasonCode] = mapped_column(
        Enum(ReasonCode, name="reason_code", values_callable=enum_values),
        nullable=False,
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_options: Mapped[dict] = mapped_column(JSONB, nullable=False)
    free_text_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    user_answer: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    session: Mapped["ClarificationSession"] = relationship(back_populates="questions")
