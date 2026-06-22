"""Bounded Semantic Repair for FinFlow's semantic pipeline.

Applies typed patch sets against declared draft paths to fix structural
failures reported by the Coverage Validator. Repair is bounded to a maximum
of one attempt per pipeline invocation and only produces typed patches
(add, replace, remove) — never a complete re-extraction.

Key constraints:
- Only accepts patch paths declared in Coverage_Validator structural failures
- Maximum one repair attempt per pipeline invocation
- Returns SemanticPatch list (typed operations only)
- Logs declared patch path, input failure, operation type, resulting modification

Requirements: 4.1, 4.2, 4.3, 4.6
"""

from __future__ import annotations

import logging
from typing import Any

from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    SemanticResolver,
)
from finflow_agent.models.draft import SemanticIntentDraft
from finflow_agent.models.patches import PatchOp, SemanticPatch
from finflow_agent.models.provenance import ProvenanceRef
from finflow_agent.pipeline.coverage_validator import StructuralFailure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RepairAlreadyAttemptedError(Exception):
    """Raised when a second repair attempt is made within the same pipeline invocation.

    The bounded repair contract permits exactly one repair attempt per invocation.
    Subsequent attempts must fail closed rather than retrying.

    Requirements: 4.2
    """

    def __init__(self) -> None:
        super().__init__(
            "Semantic repair has already been attempted in this pipeline invocation. "
            "Maximum one attempt is permitted (Requirement 4.2)."
        )


# ---------------------------------------------------------------------------
# SemanticRepair implementation
# ---------------------------------------------------------------------------


class SemanticRepair:
    """Bounded semantic repair that produces typed patches for structural failures.

    Accepts only declared patch paths from Coverage_Validator failures, performs
    at most one repair attempt per pipeline invocation, and returns typed
    SemanticPatch operations (add/replace/remove only).

    Requirements: 4.1, 4.2, 4.3, 4.6
    """

    def __init__(self, resolver: SemanticResolver) -> None:
        """Initialize SemanticRepair with an LLM adapter.

        Args:
            resolver: The LLM adapter used to generate patch proposals.
        """
        self._resolver = resolver
        self._attempted: bool = False

    @property
    def attempted(self) -> bool:
        """Whether a repair attempt has already been made in this invocation."""
        return self._attempted

    def reset(self) -> None:
        """Reset repair state for a new pipeline invocation.

        Must be called at the start of each pipeline invocation to allow
        a fresh repair attempt.
        """
        self._attempted = False
        logger.debug("Semantic repair state reset for new pipeline invocation")

    async def repair(
        self,
        draft: SemanticIntentDraft,
        failures: list[StructuralFailure],
        prompt: str,
    ) -> list[SemanticPatch]:
        """Produce typed patches to fix declared structural failures.

        Args:
            draft: The current SemanticIntentDraft to repair.
            failures: Structural failures from Coverage_Validator.
            prompt: The original user prompt for context.

        Returns:
            List of SemanticPatch operations (add/replace/remove only).

        Raises:
            RepairAlreadyAttemptedError: If repair has already been attempted
                in this pipeline invocation.

        Requirements: 4.1, 4.2, 4.3, 4.6
        """
        # Enforce maximum one attempt per invocation (Req 4.2)
        if self._attempted:
            raise RepairAlreadyAttemptedError()

        self._attempted = True

        if not failures:
            logger.info("No structural failures to repair; returning empty patch list")
            return []

        # Extract declared patch paths from validator failures (Req 4.1)
        declared_paths = {failure.element_path for failure in failures}

        logger.info(
            "Starting semantic repair attempt",
            extra={
                "failure_count": len(failures),
                "declared_paths": sorted(declared_paths),
            },
        )

        # Log each input failure (Req 4.6)
        for failure in failures:
            logger.info(
                "Input failure for repair",
                extra={
                    "patch_path": failure.element_path,
                    "failure_category": failure.category.value,
                    "failure_description": failure.description,
                },
            )

        # Build LLM prompt for patch generation
        messages = self._build_repair_messages(draft, failures, prompt)

        # Call LLM with repair-specific constraints
        constraint = DEFAULT_CONSTRAINTS[LLMCallSite.REPAIR]
        response = await self._resolver.call(
            messages,
            call_site=LLMCallSite.REPAIR,
            constraint=constraint,
        )

        # Parse and validate patches from LLM response
        patches = self._parse_patches(response.parsed, declared_paths, failures)

        # Log resulting patches (Req 4.6)
        for patch in patches:
            logger.info(
                "Semantic repair patch produced",
                extra={
                    "patch_path": patch.path,
                    "operation": patch.operation.value,
                    "source_failure": patch.source_failure,
                    "reason": patch.reason,
                    "has_value": patch.value is not None,
                },
            )

        logger.info(
            "Semantic repair attempt completed",
            extra={
                "patches_produced": len(patches),
                "declared_paths_addressed": len(
                    {p.path for p in patches} & declared_paths
                ),
            },
        )

        return patches

    def _build_repair_messages(
        self,
        draft: SemanticIntentDraft,
        failures: list[StructuralFailure],
        prompt: str,
    ) -> list[dict[str, str]]:
        """Build the LLM messages for repair patch generation.

        Structures the request to guide the LLM toward producing only typed
        patches against declared paths.
        """
        failure_descriptions = "\n".join(
            f"- Path: {f.element_path} | Category: {f.category.value} | "
            f"Description: {f.description}"
            for f in failures
        )

        declared_paths = sorted({f.element_path for f in failures})
        paths_list = "\n".join(f"- {p}" for p in declared_paths)

        draft_json = draft.model_dump(mode="json")

        system_message = (
            "You are a semantic repair agent. Your task is to produce typed patches "
            "(add, replace, or remove operations) to fix structural failures in a "
            "SemanticIntentDraft. You must ONLY produce patches against the declared "
            "paths listed below. Do NOT produce a complete re-extraction.\n\n"
            "Permitted operations: add, replace, remove\n"
            f"Declared patch paths:\n{paths_list}\n\n"
            "Return a JSON array of patch objects, each with:\n"
            '- "operation": one of "add", "replace", "remove"\n'
            '- "path": must be one of the declared paths above\n'
            '- "value": the new value (required for add/replace, omit for remove)\n'
            '- "reason": explanation of why this patch fixes the failure\n'
            '- "source_failure": which failure category this addresses\n'
        )

        user_message = (
            f"Original prompt: {prompt}\n\n"
            f"Current draft (JSON):\n{draft_json}\n\n"
            f"Structural failures to repair:\n{failure_descriptions}\n\n"
            "Produce patches to fix these failures. Only use the declared paths."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _parse_patches(
        self,
        parsed_response: Any,
        declared_paths: set[str],
        failures: list[StructuralFailure],
    ) -> list[SemanticPatch]:
        """Parse and validate patches from LLM response.

        Filters out any patches that target undeclared paths or use
        invalid operations (Req 4.1, 4.3).

        Args:
            parsed_response: The parsed JSON from the LLM response (dict or None).
            declared_paths: Set of valid patch paths from validator failures.
            failures: The original structural failures for source attribution.

        Returns:
            Validated list of SemanticPatch objects.
        """
        patches: list[SemanticPatch] = []

        if parsed_response is None:
            logger.warning("LLM response parsed field is None; returning empty patches")
            return []

        # Handle both dict with "patches" key and direct list
        raw_patches: list[dict[str, Any]]
        if isinstance(parsed_response, list):
            raw_patches = parsed_response
        elif isinstance(parsed_response, dict) and "patches" in parsed_response:
            raw_patches = parsed_response["patches"]
        elif isinstance(parsed_response, dict):
            # Try to interpret the dict as containing a list at some key
            for value in parsed_response.values():
                if isinstance(value, list):
                    raw_patches = value
                    break
            else:
                logger.warning(
                    "LLM response did not contain a parseable patch list",
                    extra={"response_type": type(parsed_response).__name__},
                )
                return []
        else:
            logger.warning(
                "LLM response format not recognized for patch extraction",
                extra={"response_type": type(parsed_response).__name__},
            )
            return []

        # Build a failure lookup for source attribution
        failure_by_path: dict[str, StructuralFailure] = {
            f.element_path: f for f in failures
        }

        for raw_patch in raw_patches:
            if not isinstance(raw_patch, dict):
                logger.warning("Skipping non-dict patch entry")
                continue

            # Validate operation type (Req 4.3: only add/replace/remove)
            op_str = raw_patch.get("operation", "").lower()
            try:
                operation = PatchOp(op_str)
            except ValueError:
                logger.warning(
                    "Rejecting patch with invalid operation",
                    extra={"operation": op_str},
                )
                continue

            # Validate path is among declared paths (Req 4.1)
            path = raw_patch.get("path", "")
            if path not in declared_paths:
                logger.warning(
                    "Rejecting patch targeting undeclared path",
                    extra={"path": path, "declared_paths": sorted(declared_paths)},
                )
                continue

            # Extract value (required for add/replace, None for remove)
            value = raw_patch.get("value")
            if operation in (PatchOp.ADD, PatchOp.REPLACE) and value is None:
                logger.warning(
                    "Rejecting add/replace patch without value",
                    extra={"path": path, "operation": operation.value},
                )
                continue

            reason = raw_patch.get("reason", "No reason provided")
            source_failure = raw_patch.get(
                "source_failure",
                failure_by_path.get(path, failures[0]).category.value
                if failures
                else "unknown",
            )

            patch = SemanticPatch(
                operation=operation,
                path=path,
                value=value if operation != PatchOp.REMOVE else None,
                reason=reason,
                provenance=[],
                source_failure=source_failure,
            )
            patches.append(patch)

        return patches
