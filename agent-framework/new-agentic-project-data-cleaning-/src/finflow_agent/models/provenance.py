"""Provenance models for FinFlow's semantic pipeline.

Defines typed provenance references that trace every semantic element back to its
source: raw prompt spans, user clarification responses, or schema evidence.

ProvenanceRef is a discriminated union (discriminator="type") ensuring unambiguous
deserialization. All models use strict configuration to prevent silent coercion.

Requirements: 1.6, 14.6, 15.1, 15.5
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptSpanProvenance(BaseModel):
    """Provenance from a span in the original user prompt.

    Records the Unicode code-point offsets and extracted text for a semantic
    element sourced directly from the raw prompt.

    Requirements: 15.1 - start offset, end offset, source text for every
    extracted action, reference, operator, value, logical group, and exclusion.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["prompt_span"] = "prompt_span"
    start_offset: int = Field(
        ..., ge=0, description="Unicode code-point offset of span start in original prompt"
    )
    end_offset: int = Field(
        ..., description="Unicode code-point offset of span end (exclusive) in original prompt"
    )
    source_text: str = Field(
        ..., min_length=1, description="Extracted text from the prompt span"
    )

    @model_validator(mode="after")
    def _validate_offsets(self) -> "PromptSpanProvenance":
        """Ensure start_offset < end_offset."""
        if self.start_offset >= self.end_offset:
            raise ValueError(
                f"start_offset ({self.start_offset}) must be less than "
                f"end_offset ({self.end_offset})"
            )
        return self


class ClarificationProvenance(BaseModel):
    """Provenance from a user clarification response.

    Records the question/response identifiers and selected value when a semantic
    element is sourced from an interactive clarification session.

    Requirements: 15.5 - question_id, response_id, selected_value (not synthetic
    prompt spans).
    """

    model_config = ConfigDict(strict=True)

    type: Literal["clarification"] = "clarification"
    question_id: str = Field(
        ..., min_length=1, description="Identifier of the clarification question"
    )
    response_id: str = Field(
        ..., min_length=1, description="Identifier of the user's response"
    )
    selected_value: str = Field(
        ..., min_length=1, description="The value selected by the user"
    )


class SchemaEvidenceProvenance(BaseModel):
    """Provenance from schema evidence.

    Records when a semantic element is inferred from schema structure or data
    evidence rather than explicit user input.

    Requirements: 15.4 - schema_fingerprint, column, evidence list for elements
    inferred from schema evidence.
    """

    model_config = ConfigDict(strict=True)

    type: Literal["schema_evidence"] = "schema_evidence"
    schema_fingerprint: str = Field(
        ..., min_length=1, description="Fingerprint of the schema used as evidence source"
    )
    column: str = Field(
        ..., min_length=1, description="Column name that provided the evidence"
    )
    evidence: list[str] = Field(
        ..., min_length=1, description="List of evidence items supporting this provenance"
    )


# Discriminated union of all provenance types.
# Uses Pydantic's discriminator on the "type" field for unambiguous deserialization.
ProvenanceRef = Annotated[
    Union[PromptSpanProvenance, ClarificationProvenance, SchemaEvidenceProvenance],
    Field(discriminator="type"),
]
