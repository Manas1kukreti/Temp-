"""Pydantic API schemas for the interactive ambiguity resolution subsystem.

These schemas define the request/response payloads for the clarification API
endpoint (POST /api/uploads/{submission_id}/clarify) and WebSocket event payloads.

Requirements: 4.1, 4.2, 5.4
"""

from __future__ import annotations

import enum
from uuid import UUID

from pydantic import BaseModel, Field


class ClarificationOutcome(str, enum.Enum):
    """Possible outcomes of processing a clarification response."""

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


class ClarificationAnswer(BaseModel):
    """A single answer to a clarification question.

    The user provides either a selected_option from the candidate list,
    or free_text when selecting 'none_of_these'.
    """

    question_id: UUID
    selected_option: str | None = None
    free_text: str | None = None


class ClarificationResponsePayload(BaseModel):
    """Request payload for POST /api/uploads/{submission_id}/clarify.

    Contains the user's answers to clarification questions along with
    concurrency control fields (session_id, intent_version, revision_token).
    """

    session_id: UUID
    intent_version: int
    revision_token: str | None = None
    answers: list[ClarificationAnswer]


class QuestionError(BaseModel):
    """Per-question error detail returned when validation fails."""

    question_id: UUID
    reason: str


class ClarificationQuestionSchema(BaseModel):
    """Serialization schema for clarification questions sent to the frontend.

    Represents a single structured question targeting an unresolved intent path.
    """

    id: UUID
    session_id: UUID
    round_number: int
    intent_path: str = Field(
        description="JSON-path to the unresolved field in the CanonicalIntent"
    )
    reason_code: ReasonCode
    question_text: str
    candidate_options: list[str] = Field(default_factory=list)
    free_text_enabled: bool = True
    user_answer: dict | None = None


class ClarificationOutcomeResponse(BaseModel):
    """Response payload from POST /api/uploads/{submission_id}/clarify.

    Communicates the result of processing the user's clarification answers,
    including any per-question errors or remaining questions for further rounds.
    """

    outcome: ClarificationOutcome
    submission_id: UUID
    session_id: UUID
    intent_version: int
    error_details: list[QuestionError] | None = None
    remaining_questions: list[ClarificationQuestionSchema] | None = None
