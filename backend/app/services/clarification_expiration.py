"""
ExpirationScheduler — periodic background task for clarification session cleanup.

Queries all expired-but-active clarification sessions and transitions them
to the expired state, quarantining their associated submissions.

Requirements: 10.1, 10.2, 10.4
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.clarification_session_repo import SessionRepository
from app.services.clarification_service import ClarificationService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLEANUP_INTERVAL_SECONDS: int = 60
"""How often (in seconds) the background cleanup loop runs."""


# ---------------------------------------------------------------------------
# ExpirationScheduler
# ---------------------------------------------------------------------------


class ExpirationScheduler:
    """Periodic background task that expires stale clarification sessions.

    Can be instantiated with a database session for direct invocation (e.g.,
    in tests), or run as a long-lived asyncio background task via
    `start_cleanup_loop` / `stop_cleanup_loop`.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def run_cleanup(self) -> int:
        """Find and expire all sessions past their expires_at with status=active.

        For each expired session, delegates to ClarificationService.expire_session()
        which handles status transition, submission quarantine, and WebSocket broadcast.

        Returns:
            The number of expired sessions processed.
        """
        repo = SessionRepository(self.db)
        service = ClarificationService(self.db)

        expired_sessions = await repo.get_expired_sessions()

        if not expired_sessions:
            return 0

        count = 0
        for session in expired_sessions:
            try:
                await service.expire_session(session.id)
                count += 1
            except Exception:
                logger.exception(
                    "Failed to expire session %s for submission %s",
                    session.id,
                    session.submission_id,
                )

        # Commit all changes in one transaction
        await self.db.commit()

        if count > 0:
            logger.info(
                "ExpirationScheduler: expired %d clarification session(s)", count
            )

        return count

    async def __call__(self) -> int:
        """Allow the scheduler to be called as an async function (for testing)."""
        return await self.run_cleanup()


# ---------------------------------------------------------------------------
# Background loop management (asyncio periodic task)
# ---------------------------------------------------------------------------

_cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    """Internal loop that periodically runs session cleanup."""
    from app.db.session import AsyncSessionLocal

    while True:
        try:
            async with AsyncSessionLocal() as db:
                scheduler = ExpirationScheduler(db)
                await scheduler.run_cleanup()
        except asyncio.CancelledError:
            logger.info("ExpirationScheduler background loop cancelled.")
            break
        except Exception:
            logger.exception("ExpirationScheduler loop encountered an error.")

        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


async def start_cleanup_loop(app=None) -> None:
    """Start the periodic expiration cleanup as a background asyncio task.

    Typically called from the FastAPI startup event. Stores the task reference
    on `app.state` if an app is provided, or in the module-level `_cleanup_task`.
    """
    global _cleanup_task
    task = asyncio.create_task(_cleanup_loop(), name="clarification_expiration_cleanup")
    _cleanup_task = task

    if app is not None:
        app.state.clarification_cleanup_task = task

    logger.info(
        "ExpirationScheduler started (interval=%ds)", CLEANUP_INTERVAL_SECONDS
    )


async def stop_cleanup_loop(app=None) -> None:
    """Cancel the periodic expiration cleanup task.

    Typically called from the FastAPI shutdown event.
    """
    global _cleanup_task

    task = None
    if app is not None:
        task = getattr(app.state, "clarification_cleanup_task", None)

    if task is None:
        task = _cleanup_task

    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _cleanup_task = None
    if app is not None:
        app.state.clarification_cleanup_task = None

    logger.info("ExpirationScheduler stopped.")
