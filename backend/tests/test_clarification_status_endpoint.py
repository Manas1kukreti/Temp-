"""
Unit tests for GET /api/uploads/{submission_id}/clarification-status endpoint.

Tests the REST polling endpoint used by the frontend to resync
clarification session state after a WebSocket disconnect.

Requirements: Error handling graceful degradation
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.clarification import get_clarification_status
from app.models.clarification import ReasonCode, SessionStatus


def _make_question(
    question_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    round_number: int = 1,
    intent_path: str = "actions[0].conditions[0].field.raw_reference",
    reason_code: ReasonCode = ReasonCode.MULTIPLE_COLUMN_MATCHES,
    question_text: str = "Which column did you mean?",
    candidate_options: list[str] | None = None,
    free_text_enabled: bool = True,
    user_answer: dict | None = None,
) -> MagicMock:
    """Create a mock ClarificationQuestion."""
    q = MagicMock()
    q.id = question_id or uuid.uuid4()
    q.session_id = session_id or uuid.uuid4()
    q.round_number = round_number
    q.intent_path = intent_path
    q.reason_code = reason_code
    q.question_text = question_text
    q.candidate_options = candidate_options or ["col_a", "col_b", "none_of_these"]
    q.free_text_enabled = free_text_enabled
    q.user_answer = user_answer
    return q


def _make_session(
    session_id: uuid.UUID | None = None,
    submission_id: uuid.UUID | None = None,
    intent_version: int = 1,
    round_count: int = 0,
    max_rounds: int = 2,
    revision_token: str = "test-revision-token-abc123xyz456",
    status: SessionStatus = SessionStatus.active,
    expires_at: datetime | None = None,
    questions: list | None = None,
) -> MagicMock:
    """Create a mock ClarificationSession."""
    session = MagicMock()
    session.id = session_id or uuid.uuid4()
    session.submission_id = submission_id or uuid.uuid4()
    session.intent_version = intent_version
    session.round_count = round_count
    session.max_rounds = max_rounds
    session.revision_token = revision_token
    session.status = status
    session.expires_at = expires_at or (datetime.now(UTC) + timedelta(minutes=30))
    session.questions = questions or []
    return session


class TestGetClarificationStatus:
    """Tests for GET /{submission_id}/clarification-status endpoint."""

    @pytest.mark.asyncio
    async def test_returns_404_when_no_active_session(self):
        """No active session returns 404 with appropriate message."""
        submission_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_user = MagicMock()

        with patch(
            "app.api.clarification.SessionRepository"
        ) as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.get_active_by_submission = AsyncMock(return_value=None)

            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await get_clarification_status(
                    submission_id=submission_id,
                    db=mock_db,
                    user=mock_user,
                )

            assert exc_info.value.status_code == 404
            assert "No active clarification session" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_session_state_when_active(self):
        """Active session returns full session state with 200."""
        submission_id = uuid.uuid4()
        session_id = uuid.uuid4()
        q_id = uuid.uuid4()
        expires_at = datetime.now(UTC) + timedelta(minutes=25)

        question = _make_question(
            question_id=q_id,
            session_id=session_id,
            round_number=1,
            intent_path="actions[0].field.raw_reference",
            reason_code=ReasonCode.MULTIPLE_COLUMN_MATCHES,
            question_text="Which column did you mean by 'amount'?",
            candidate_options=["total_amount", "net_amount", "none_of_these"],
        )

        session = _make_session(
            session_id=session_id,
            submission_id=submission_id,
            intent_version=2,
            round_count=1,
            max_rounds=2,
            revision_token="rev-token-xyz",
            expires_at=expires_at,
            questions=[question],
        )

        mock_db = AsyncMock()
        mock_user = MagicMock()

        with patch(
            "app.api.clarification.SessionRepository"
        ) as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.get_active_by_submission = AsyncMock(return_value=session)

            response = await get_clarification_status(
                submission_id=submission_id,
                db=mock_db,
                user=mock_user,
            )

        assert response.status_code == 200
        content = response.body.decode()
        import json

        data = json.loads(content)

        assert data["session_id"] == str(session_id)
        assert data["submission_id"] == str(submission_id)
        assert data["intent_version"] == 2
        assert data["round_count"] == 1
        assert data["max_rounds"] == 2
        assert data["revision_token"] == "rev-token-xyz"
        assert data["expires_at"] == expires_at.isoformat()
        assert data["status"] == "active"
        assert len(data["questions"]) == 1

        q_data = data["questions"][0]
        assert q_data["id"] == str(q_id)
        assert q_data["session_id"] == str(session_id)
        assert q_data["round_number"] == 1
        assert q_data["intent_path"] == "actions[0].field.raw_reference"
        assert q_data["reason_code"] == "MULTIPLE_COLUMN_MATCHES"
        assert q_data["question_text"] == "Which column did you mean by 'amount'?"
        assert q_data["candidate_options"] == ["total_amount", "net_amount", "none_of_these"]
        assert q_data["free_text_enabled"] is True
        assert q_data["user_answer"] is None

    @pytest.mark.asyncio
    async def test_returns_multiple_questions(self):
        """Session with multiple questions returns all of them."""
        submission_id = uuid.uuid4()
        session_id = uuid.uuid4()

        questions = [
            _make_question(session_id=session_id, question_text=f"Question {i}")
            for i in range(3)
        ]

        session = _make_session(
            session_id=session_id,
            submission_id=submission_id,
            questions=questions,
        )

        mock_db = AsyncMock()
        mock_user = MagicMock()

        with patch(
            "app.api.clarification.SessionRepository"
        ) as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.get_active_by_submission = AsyncMock(return_value=session)

            response = await get_clarification_status(
                submission_id=submission_id,
                db=mock_db,
                user=mock_user,
            )

        assert response.status_code == 200
        import json

        data = json.loads(response.body.decode())
        assert len(data["questions"]) == 3

    @pytest.mark.asyncio
    async def test_handles_session_with_no_questions(self):
        """Session with empty questions list returns empty array."""
        submission_id = uuid.uuid4()
        session_id = uuid.uuid4()

        session = _make_session(
            session_id=session_id,
            submission_id=submission_id,
            questions=[],
        )

        mock_db = AsyncMock()
        mock_user = MagicMock()

        with patch(
            "app.api.clarification.SessionRepository"
        ) as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.get_active_by_submission = AsyncMock(return_value=session)

            response = await get_clarification_status(
                submission_id=submission_id,
                db=mock_db,
                user=mock_user,
            )

        assert response.status_code == 200
        import json

        data = json.loads(response.body.decode())
        assert data["questions"] == []

    @pytest.mark.asyncio
    async def test_handles_null_expires_at(self):
        """Session with None expires_at returns null in response."""
        submission_id = uuid.uuid4()
        session_id = uuid.uuid4()

        session = _make_session(
            session_id=session_id,
            submission_id=submission_id,
            expires_at=None,
            questions=[],
        )
        # Override the default expires_at set by _make_session
        session.expires_at = None

        mock_db = AsyncMock()
        mock_user = MagicMock()

        with patch(
            "app.api.clarification.SessionRepository"
        ) as MockRepo:
            mock_repo_instance = MockRepo.return_value
            mock_repo_instance.get_active_by_submission = AsyncMock(return_value=session)

            response = await get_clarification_status(
                submission_id=submission_id,
                db=mock_db,
                user=mock_user,
            )

        assert response.status_code == 200
        import json

        data = json.loads(response.body.decode())
        assert data["expires_at"] is None
