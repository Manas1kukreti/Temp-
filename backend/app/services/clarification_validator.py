"""
Response validator for the interactive ambiguity resolution subsystem.

Validates user-submitted clarification answers before they are applied
as intent patches. Checks revision tokens, intent versions, question
ownership, option validity, and free-text requirements.

Requirements: 4.3, 4.4, 4.5, 5.1, 5.2, 5.4, 9.2, 9.3, 9.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.models.clarification import ClarificationSession
from app.schemas.clarification import ClarificationResponsePayload


@dataclass
class ValidationResult:
    """Result of validating a clarification response.

    Attributes:
        is_valid: Whether the response passed all validation checks.
        rejection_type: Category of the first structural rejection encountered.
            One of: "token_mismatch", "version_mismatch", "invalid_question_id",
            "invalid_answers", or None if valid.
        error_details: Per-question error information for INVALID_RESPONSE cases.
            Each dict contains "question_id" (str) and "reason" (str).
    """

    is_valid: bool = True
    rejection_type: str | None = None
    error_details: list[dict] | None = None


class ResponseValidator:
    """Validates clarification responses against session state.

    Performs the following checks in order:
    1. Revision token match (skipped if token is None per Req 9.4)
    2. Intent version match
    3. All question_ids belong to the current session
    4. Each answer's selected_option is valid for its question
    5. "none_of_these" answers include non-empty free_text
    """

    def validate(
        self,
        session: ClarificationSession,
        response: ClarificationResponsePayload,
    ) -> ValidationResult:
        """Validate a clarification response against the current session state.

        Args:
            session: The active ClarificationSession with loaded questions.
            response: The user-submitted clarification response payload.

        Returns:
            ValidationResult indicating whether the response is valid,
            and if not, the rejection type and per-question error details.
        """
        # --- Check 1: Revision token (Req 9.2, 9.3, 9.4) ---
        # If the response provides a revision_token, it must match the session's
        # current token. If None/absent, skip token validation (Req 9.4).
        if response.revision_token is not None:
            if response.revision_token != session.revision_token:
                return ValidationResult(
                    is_valid=False,
                    rejection_type="token_mismatch",
                )

        # --- Check 2: Intent version (Req 4.4) ---
        if response.intent_version != session.intent_version:
            return ValidationResult(
                is_valid=False,
                rejection_type="version_mismatch",
            )

        # --- Build question lookup from session ---
        # Map question_id -> ClarificationQuestion for ownership and option checks
        session_question_ids: dict[UUID, object] = {
            q.id: q for q in session.questions
        }

        # --- Check 3: All question_ids belong to current session (Req 4.5) ---
        for answer in response.answers:
            if answer.question_id not in session_question_ids:
                return ValidationResult(
                    is_valid=False,
                    rejection_type="invalid_question_id",
                    error_details=[
                        {
                            "question_id": str(answer.question_id),
                            "reason": "Question does not belong to the current session.",
                        }
                    ],
                )

        # --- Checks 4 & 5: Per-answer validation (Req 5.1, 5.2, 5.4) ---
        answer_errors: list[dict] = []

        for answer in response.answers:
            question = session_question_ids[answer.question_id]

            # Get the list of valid candidate options from the question
            candidate_options = _get_candidate_options(question)

            selected = answer.selected_option

            if selected is not None:
                # Check 4: selected_option must be in candidate_options or "none_of_these"
                if selected != "none_of_these" and selected not in candidate_options:
                    answer_errors.append(
                        {
                            "question_id": str(answer.question_id),
                            "reason": (
                                f"Selected option '{selected}' is not a valid "
                                f"candidate for this question."
                            ),
                        }
                    )
                    continue

                # Check 5: "none_of_these" requires non-empty free_text
                if selected == "none_of_these":
                    free_text = (answer.free_text or "").strip()
                    if not free_text:
                        answer_errors.append(
                            {
                                "question_id": str(answer.question_id),
                                "reason": (
                                    "Selecting 'none_of_these' requires a non-empty "
                                    "free_text value."
                                ),
                            }
                        )

        if answer_errors:
            return ValidationResult(
                is_valid=False,
                rejection_type="invalid_answers",
                error_details=answer_errors,
            )

        return ValidationResult(is_valid=True)


def _get_candidate_options(question) -> list[str]:
    """Extract the list of candidate option strings from a ClarificationQuestion.

    The candidate_options field in the database is stored as JSONB.
    It may be a list of strings directly, or a list of dicts with a 'value' key.
    This helper normalizes both representations.
    """
    raw = question.candidate_options
    if not raw:
        return []
    if isinstance(raw, list):
        # If elements are strings, return as-is
        if all(isinstance(opt, str) for opt in raw):
            return raw
        # If elements are dicts, extract the 'value' key
        return [opt.get("value", "") if isinstance(opt, dict) else str(opt) for opt in raw]
    return []
