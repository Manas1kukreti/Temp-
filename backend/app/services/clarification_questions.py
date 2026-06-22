"""
Question generation for the interactive ambiguity resolution subsystem.

Produces structured ClarificationQuestion objects from unresolved fields in
the CanonicalIntent. Each question targets a specific unresolved intent path
and provides candidate options based on the reason the field is unresolved.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 14.1, 14.2, 14.3
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.models.clarification import ReasonCode


@dataclass
class UnresolvedField:
    """Represents a single unresolved field in the CanonicalIntent.

    Attributes:
        intent_path: JSON-path to the unresolved field
            (e.g., "actions[1].conditions[0].field.raw_reference").
        reason_code: Classification of why the field is unresolved.
        raw_reference: The original user-facing term that could not be resolved.
        grounding_candidates: Candidate columns with confidence scores,
            each dict has 'column_name' (str) and 'confidence' (float).
    """

    intent_path: str
    reason_code: ReasonCode
    raw_reference: str
    grounding_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratedQuestion:
    """A generated clarification question ready for persistence.

    This is a lightweight data transfer object used before the question
    is persisted as a ClarificationQuestion database model.
    """

    question_id: str
    intent_path: str
    reason_code: ReasonCode
    question_text: str
    candidate_options: list[str]
    free_text_enabled: bool = True


class QuestionGenerator:
    """Generates structured ClarificationQuestion objects from unresolved fields.

    For each unresolved field, produces exactly one question with:
    - A human-readable question_text tailored to the reason_code
    - candidate_options derived from grounding candidates (ordered by confidence desc)
    - "none_of_these" always appended as the final candidate option
    - free_text_enabled set to True
    """

    def generate(
        self,
        unresolved_fields: list[UnresolvedField],
        intent_package: Any,
        intent: Any,
    ) -> list[GeneratedQuestion]:
        """Produce one ClarificationQuestion per unresolved field.

        Args:
            unresolved_fields: List of fields that could not be resolved.
            intent_package: The IntentPackage schema-resolution artifact.
            intent: The CanonicalIntent structured representation.

        Returns:
            A list of GeneratedQuestion objects, one per unresolved field.
        """
        questions: list[GeneratedQuestion] = []

        for field in unresolved_fields:
            question = self._generate_question(field, intent_package, intent)
            questions.append(question)

        return questions

    def _generate_question(
        self,
        field: UnresolvedField,
        intent_package: Any,
        intent: Any,
    ) -> GeneratedQuestion:
        """Generate a single question for an unresolved field based on its reason code."""
        reason = field.reason_code

        if reason == ReasonCode.MULTIPLE_COLUMN_MATCHES:
            return self._question_for_multiple_matches(field)
        elif reason == ReasonCode.AMBIGUOUS_REFERENCE:
            return self._question_for_ambiguous_reference(field)
        elif reason == ReasonCode.LOW_CONFIDENCE_SCORE:
            return self._question_for_low_confidence(field)
        elif reason == ReasonCode.MISSING_COLUMN:
            return self._question_for_missing_column(field)
        elif reason == ReasonCode.CONFLICTING_EVIDENCE:
            return self._question_for_conflicting_evidence(field)
        else:
            # Fallback for any unexpected reason code
            return self._question_for_low_confidence(field)

    def _question_for_multiple_matches(
        self, field: UnresolvedField
    ) -> GeneratedQuestion:
        """MULTIPLE_COLUMN_MATCHES: present candidates ordered by confidence descending.

        The question_text only presents the options without describing the
        column matching algorithm (Requirement 2.4).
        """
        candidates = self._sorted_candidates(field.grounding_candidates)
        candidate_options = [c["column_name"] for c in candidates]
        candidate_options.append("none_of_these")

        options_list = ", ".join(f"'{opt}'" for opt in candidate_options[:-1])
        question_text = (
            f"Which column does '{field.raw_reference}' refer to? "
            f"Options: {options_list}"
        )

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            intent_path=field.intent_path,
            reason_code=field.reason_code,
            question_text=question_text,
            candidate_options=candidate_options,
            free_text_enabled=True,
        )

    def _question_for_ambiguous_reference(
        self, field: UnresolvedField
    ) -> GeneratedQuestion:
        """AMBIGUOUS_REFERENCE: describe ambiguity with competing interpretations as options.

        The question_text describes the ambiguity and presents the competing
        interpretations as candidate_options (Requirement 2.5).
        """
        candidates = self._sorted_candidates(field.grounding_candidates)
        candidate_options = [c["column_name"] for c in candidates]
        candidate_options.append("none_of_these")

        interpretations = ", ".join(f"'{opt}'" for opt in candidate_options[:-1])
        question_text = (
            f"The reference '{field.raw_reference}' is ambiguous. "
            f"It could refer to multiple interpretations: {interpretations}. "
            f"Which interpretation did you mean?"
        )

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            intent_path=field.intent_path,
            reason_code=field.reason_code,
            question_text=question_text,
            candidate_options=candidate_options,
            free_text_enabled=True,
        )

    def _question_for_low_confidence(self, field: UnresolvedField) -> GeneratedQuestion:
        """LOW_CONFIDENCE_SCORE: ask which column the user meant with top candidates."""
        candidates = self._sorted_candidates(field.grounding_candidates)
        candidate_options = [c["column_name"] for c in candidates]
        candidate_options.append("none_of_these")

        question_text = f"Which column did you mean by '{field.raw_reference}'?"

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            intent_path=field.intent_path,
            reason_code=field.reason_code,
            question_text=question_text,
            candidate_options=candidate_options,
            free_text_enabled=True,
        )

    def _question_for_missing_column(self, field: UnresolvedField) -> GeneratedQuestion:
        """MISSING_COLUMN: prompt user for the correct column name via free text.

        Even though the primary response mechanism is free text, we still
        provide "none_of_these" as the final candidate option per Requirement 2.3.
        """
        candidate_options = ["none_of_these"]

        question_text = (
            f"The column '{field.raw_reference}' was not found in the dataset. "
            f"Please provide the correct column name."
        )

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            intent_path=field.intent_path,
            reason_code=field.reason_code,
            question_text=question_text,
            candidate_options=candidate_options,
            free_text_enabled=True,
        )

    def _question_for_conflicting_evidence(
        self, field: UnresolvedField
    ) -> GeneratedQuestion:
        """CONFLICTING_EVIDENCE: describe the conflict and ask for disambiguation."""
        candidates = self._sorted_candidates(field.grounding_candidates)
        candidate_options = [c["column_name"] for c in candidates]
        candidate_options.append("none_of_these")

        conflicting_columns = ", ".join(f"'{opt}'" for opt in candidate_options[:-1])
        question_text = (
            f"There is conflicting evidence for '{field.raw_reference}'. "
            f"The following columns have contradictory signals: {conflicting_columns}. "
            f"Which column should be used?"
        )

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            intent_path=field.intent_path,
            reason_code=field.reason_code,
            question_text=question_text,
            candidate_options=candidate_options,
            free_text_enabled=True,
        )

    @staticmethod
    def _sorted_candidates(
        grounding_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort grounding candidates by confidence score descending.

        Args:
            grounding_candidates: List of dicts with 'column_name' and 'confidence' keys.

        Returns:
            A new list sorted by confidence descending.
        """
        return sorted(
            grounding_candidates,
            key=lambda c: c.get("confidence", 0.0),
            reverse=True,
        )
