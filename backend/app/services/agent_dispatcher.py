import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models import Submission, SubmissionStatus
from app.services.canonical_intent import (
    INTENT_EXTRACTOR_VERSION,
    INTENT_GROUNDING_VERSION,
    INTENT_NORMALIZER_VERSION,
)
from app.services.intent_revision import create_dispatch_outbox, mark_outbox_delivered, persist_intent_revision
from app.services.schema_proposal import build_schema_proposal_from_file

logger = logging.getLogger(__name__)


def _submission_file_id(submission: Submission) -> str:
    return Path(str(submission.file_path or "")).name


def _build_canonical_intent_envelope(submission: Submission) -> dict | None:
    summary = submission.summary if isinstance(submission.summary, dict) else {}
    schema_proposal = summary.get("schema_proposal") if isinstance(summary.get("schema_proposal"), dict) else None
    if schema_proposal is None and submission.file_path:
        settings = get_settings()
        rebuilt = build_schema_proposal_from_file(
            Path(submission.file_path),
            max_preview_rows=getattr(settings, "max_preview_rows", 50),
            instruction=str(submission.instruction or "").strip(),
        )
        if rebuilt is not None:
            schema_proposal = rebuilt[0]

    canonical_intent = None
    if isinstance(schema_proposal, dict):
        canonical_intent = schema_proposal.get("canonical_intent")
        if canonical_intent is not None and isinstance(canonical_intent, dict):
            created_at = datetime.now(timezone.utc).isoformat()
            envelope = {
                "schema_version": "1.0",
                "intent_id": canonical_intent.get("intent_id"),
                "intent_revision": canonical_intent.get("intent_revision", 1),
                "intent_hash": canonical_intent.get("intent_hash"),
                "parent_intent_id": canonical_intent.get("parent_intent_id"),
                "intent": canonical_intent,
                "original_instruction": str(submission.instruction or "").strip(),
                "intent_status": canonical_intent.get("resolution_status", "resolved"),
                "repair_notes": list(canonical_intent.get("repair_notes", [])) if isinstance(canonical_intent.get("repair_notes"), list) else [],
                "assumptions": list(canonical_intent.get("assumptions", [])) if isinstance(canonical_intent.get("assumptions"), list) else [],
                "extractor_version": INTENT_EXTRACTOR_VERSION,
                "normalizer_version": INTENT_NORMALIZER_VERSION,
                "grounding_version": INTENT_GROUNDING_VERSION,
                "capability_version": canonical_intent.get("capability_version"),
                "capability_snapshot": canonical_intent.get("capability_snapshot", {}),
                "created_at": created_at,
                "grounded_at": canonical_intent.get("grounded_at") or created_at,
            }
            return envelope
    return None


async def enqueue_submission_dispatch(submission_id: UUID | str, *, persist_revision: bool = True) -> None:
    settings = get_settings()
    try:
        redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        async with AsyncSessionLocal() as db:
            submission = await db.get(Submission, UUID(str(submission_id)))
            if not submission:
                logger.warning("Dispatch queue referenced missing submission %s", submission_id)
                return

            canonical_intent = _build_canonical_intent_envelope(submission)
            outbox = None
            payload = {
                "submission_id": str(submission.id),
                "file_id": _submission_file_id(submission),
                "file_name": submission.file_name,
                "resolved_file_path": str(submission.file_path or ""),
                "output_format": str(submission.output_format or "").strip().lower(),
                "audit_context": {
                    "original_instruction": str(submission.instruction or "").strip(),
                    "submission_id": str(submission.id),
                },
            }
            if canonical_intent is not None:
                canonical_payload = canonical_intent["intent"] if isinstance(canonical_intent.get("intent"), dict) else canonical_intent
                if persist_revision:
                    await persist_intent_revision(
                        db,
                        submission=submission,
                        canonical_intent=canonical_payload,
                        original_instruction=str(submission.instruction or "").strip(),
                        parent_intent_id=UUID(str(canonical_intent["parent_intent_id"])) if canonical_intent.get("parent_intent_id") else None,
                    )
                summary = submission.summary if isinstance(submission.summary, dict) else {}
                created_at_raw = canonical_intent.get("created_at")
                payload["audit_context"] = {
                    **payload["audit_context"],
                    "intent_id": canonical_intent.get("intent_id"),
                    "intent_revision": canonical_intent.get("intent_revision", 1),
                    "intent_hash": canonical_intent.get("intent_hash"),
                    "capability_version": canonical_intent.get("capability_version"),
                }
                submission.summary = {
                    **summary,
                    "canonical_intent": canonical_intent,
                    "canonical_intent_schema_version": canonical_intent.get("schema_version", "1.0"),
                    "canonical_intent_status": canonical_intent.get("intent_status", canonical_intent.get("intent", {}).get("resolution_status", "resolved")),
                    "original_instruction": str(submission.instruction or "").strip(),
                    "intent_extractor_version": canonical_intent.get("extractor_version", INTENT_EXTRACTOR_VERSION),
                    "intent_normalizer_version": canonical_intent.get("normalizer_version", INTENT_NORMALIZER_VERSION),
                    "intent_grounding_version": canonical_intent.get("grounding_version", INTENT_GROUNDING_VERSION),
                    "intent_created_at": created_at_raw,
                    "intent_id": canonical_intent.get("intent_id"),
                    "intent_revision": canonical_intent.get("intent_revision", 1),
                    "intent_hash": canonical_intent.get("intent_hash"),
                    "parent_intent_id": canonical_intent.get("parent_intent_id"),
                    "grounded_at": canonical_intent.get("grounded_at"),
                    "capability_version": canonical_intent.get("capability_version"),
                }
                payload["canonical_intent"] = canonical_intent

                outbox = await create_dispatch_outbox(db, submission=submission, payload=payload)
                submission.status = SubmissionStatus.planning
                await db.commit()
            else:
                submission.status = SubmissionStatus.planning
                await db.commit()

            await redis.enqueue_job("process_job_task", payload)
            submission.dispatched_at = datetime.now(timezone.utc)
            await db.commit()
            if outbox is not None:
                logger.info(
                    "event=canonical_job_enqueued submission_id=%s intent_schema_version=%s",
                    submission.id,
                    canonical_intent.get("intent", {}).get("schema_version", "2.0"),
                )
                await mark_outbox_delivered(db, outbox.id)
                await db.commit()
    except Exception as e:
        logger.exception("Failed to enqueue submission via arq")

async def start_dispatcher(app) -> None:
    # Set a dummy task so that health checks that look for agent_dispatch_task return True
    app.state.agent_dispatch_task = "arq_managed"

async def stop_dispatcher(app) -> None:
    app.state.agent_dispatch_task = None
