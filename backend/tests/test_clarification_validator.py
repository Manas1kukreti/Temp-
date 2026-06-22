"""
Unit tests for the ResponseValidator (clarification_validator.py).

Validates the following requirements:
- Req 4.3: Stale revision_token → rejection
- Req 4.4: Stale intent_version → rejection
- Req 4.5: Unknown question_id → rejection
- Req 5.1: Invalid selected_option → INVALID_RESPONSE
- Req 5.2: none_of_these without free_text → INVALID_RESPONSE
- Req 5.4: Per-question error_details
- Req 9.2, 9.3: Token mismatch handling
- Req 9.4: None token bypasses validation
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.schemas.clarification import ClarificationAnswer, ClarificationResponsePayload
from app.services.clarification_validator import ResponseValidator, ValidationResult


def _make_question(
    question_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    candidate_options: list[str] | None = None,
) -> MagicMock:
    """Create a mock ClarificationQuestion."""
    q = MagicMock()
    q.id = question_id or uuid.uuid4()
    q.session_id = session_id or uuid.uuid4()
    q.candidate_options = candidate_options or ["col_a", "col_b", "none_of_these"]
    return q


def _make_session(
    session_id: uuid.UUID | None = None,
    intent_version: int = 1,
    revision_token: str = "valid-token-abc123",
    questions: list | None = None,
) -> MagicMock:
    """Create a mock ClarificationSession."""
    session = MagicMock()
    session.id = session_id or uuid.uuid4()
    session.intent_version = intent_version
    session.revision_token = revision_token
    session.questions = questions or []
    session.round_count = 0
    return session


class TestResponseValidator:
    """Tests for ResponseValidator.validate()."""

    def setup_method(self):
        self.validator = ResponseValidator()

    def test_valid_response_passes(self):
        """A fully valid response returns is_valid=True."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "col_b", "none_of_these"])]
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[ClarificationAnswer(question_id=q_id, selected_option="col_a")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is True
        assert result.rejection_type is None
        assert result.error_details is None

    def test_token_mismatch_rejected(self):
        """Mismatched revision_token returns token_mismatch (Req 4.3, 9.2, 9.3)."""
        session = _make_session(revision_token="current-token")
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="stale-token",
            answers=[],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "token_mismatch"

    def test_none_token_bypasses_validation(self):
        """None revision_token skips token check (Req 9.4)."""
        q_id = uuid.uuid4()
        session = _make_session(
            revision_token="some-token",
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token=None,
            answers=[ClarificationAnswer(question_id=q_id, selected_option="col_a")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is True

    def test_version_mismatch_rejected(self):
        """Mismatched intent_version returns version_mismatch (Req 4.4)."""
        session = _make_session(intent_version=2)
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "version_mismatch"

    def test_invalid_question_id_rejected(self):
        """Unknown question_id returns invalid_question_id (Req 4.5)."""
        q_id = uuid.uuid4()
        unknown_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id)],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[ClarificationAnswer(question_id=unknown_id, selected_option="col_a")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "invalid_question_id"
        assert result.error_details is not None
        assert len(result.error_details) == 1
        assert result.error_details[0]["question_id"] == str(unknown_id)

    def test_invalid_selected_option_rejected(self):
        """selected_option not in candidate_options → invalid_answers (Req 5.1)."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "col_b", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[ClarificationAnswer(question_id=q_id, selected_option="nonexistent_column")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "invalid_answers"
        assert result.error_details is not None
        assert len(result.error_details) == 1
        assert "nonexistent_column" in result.error_details[0]["reason"]

    def test_none_of_these_without_free_text_rejected(self):
        """none_of_these without free_text → invalid_answers (Req 5.2)."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[ClarificationAnswer(question_id=q_id, selected_option="none_of_these", free_text="")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "invalid_answers"
        assert "none_of_these" in result.error_details[0]["reason"]

    def test_none_of_these_with_free_text_passes(self):
        """none_of_these with non-empty free_text is valid."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[
                ClarificationAnswer(
                    question_id=q_id,
                    selected_option="none_of_these",
                    free_text="total_revenue",
                )
            ],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is True

    def test_none_of_these_with_whitespace_only_free_text_rejected(self):
        """Whitespace-only free_text is treated as empty."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[
                ClarificationAnswer(
                    question_id=q_id,
                    selected_option="none_of_these",
                    free_text="   ",
                )
            ],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "invalid_answers"

    def test_multiple_answer_errors_collected(self):
        """Multiple invalid answers return multiple error_details (Req 5.4)."""
        q1_id = uuid.uuid4()
        q2_id = uuid.uuid4()
        session = _make_session(
            questions=[
                _make_question(question_id=q1_id, candidate_options=["col_a", "none_of_these"]),
                _make_question(question_id=q2_id, candidate_options=["col_x", "col_y", "none_of_these"]),
            ],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[
                ClarificationAnswer(question_id=q1_id, selected_option="bad_option"),
                ClarificationAnswer(question_id=q2_id, selected_option="none_of_these", free_text=""),
            ],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is False
        assert result.rejection_type == "invalid_answers"
        assert len(result.error_details) == 2

    def test_token_check_precedes_version_check(self):
        """Token mismatch is caught before version mismatch."""
        session = _make_session(intent_version=2, revision_token="current-token")
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,  # Also wrong
            revision_token="stale-token",  # Checked first
            answers=[],
        )

        result = self.validator.validate(session, response)

        assert result.rejection_type == "token_mismatch"

    def test_valid_none_of_these_is_accepted_as_option(self):
        """'none_of_these' is always a valid selected_option even if not in candidate list explicitly."""
        q_id = uuid.uuid4()
        # candidate_options does NOT include "none_of_these" explicitly
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "col_b"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[
                ClarificationAnswer(
                    question_id=q_id,
                    selected_option="none_of_these",
                    free_text="my_custom_column",
                )
            ],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is True

    def test_answer_with_no_selected_option_passes(self):
        """An answer with selected_option=None passes option validation."""
        q_id = uuid.uuid4()
        session = _make_session(
            questions=[_make_question(question_id=q_id, candidate_options=["col_a", "none_of_these"])],
        )
        response = ClarificationResponsePayload(
            session_id=session.id,
            intent_version=1,
            revision_token="valid-token-abc123",
            answers=[ClarificationAnswer(question_id=q_id, selected_option=None, free_text="something")],
        )

        result = self.validator.validate(session, response)

        assert result.is_valid is True
