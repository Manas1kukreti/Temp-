"""
WebSocketManager — channel-based WebSocket connection and broadcast management.

Manages WebSocket connections grouped by channel and provides generic broadcast
functionality for delivering real-time events to connected frontends.

Supported clarification event types (broadcast on "uploads" channel):
  - CLARIFICATION_REQUIRED: session created, questions ready for user
  - CLARIFICATION_RESOLVED: all ambiguities resolved, submission proceeds
  - CLARIFICATION_EXPIRED: session expired or max rounds exceeded
  - CLARIFICATION_ROUND_UPDATE: new round of questions after partial resolution

Requirements: 3.1, 3.2, 3.3, 12.1, 12.2, 12.3, 12.4
"""

import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clarification WebSocket Event Type Constants
# ---------------------------------------------------------------------------
# These constants define the supported clarification-related event types
# broadcast on the "uploads" channel. All clarification events MUST include
# submission_id and session_id in the payload (Req 12.4).

CLARIFICATION_REQUIRED = "clarification_required"
"""Broadcast when a new clarification session is created and questions are ready.

Expected payload schema:
    - session_id: str (UUID)
    - submission_id: str (UUID)
    - intent_version: int
    - revision_token: str
    - expires_at: str (ISO 8601 timestamp)
    - questions: list[dict] — each with question_id, intent_path, reason_code,
      question_text, candidate_options, free_text_enabled

Requirements: 3.1, 3.2
"""

CLARIFICATION_RESOLVED = "clarification_resolved"
"""Broadcast when a clarification session concludes with all ambiguities resolved.

Expected payload schema:
    - session_id: str (UUID)
    - submission_id: str (UUID)
    - intent_version: int
    - outcome: str ("RESOLVED")

Note: When outcome is RESOLVED, only this event is broadcast — no
"clarification_round_update" event should be sent (Req 12.2).

Requirements: 12.2
"""

CLARIFICATION_EXPIRED = "clarification_expired"
"""Broadcast when a clarification session expires or max rounds are exceeded.

Expected payload schema:
    - session_id: str (UUID)
    - submission_id: str (UUID)

Requirements: 3.3
"""

CLARIFICATION_ROUND_UPDATE = "clarification_round_update"
"""Broadcast when a new clarification round begins (STILL_AMBIGUOUS outcome).

Expected payload schema:
    - session_id: str (UUID)
    - submission_id: str (UUID)
    - round_count: int
    - revision_token: str
    - questions: list[dict] — each with question_id, intent_path, reason_code,
      question_text, candidate_options, free_text_enabled

Requirements: 12.3
"""

# All clarification event types for validation and documentation
CLARIFICATION_EVENT_TYPES: set[str] = {
    CLARIFICATION_REQUIRED,
    CLARIFICATION_RESOLVED,
    CLARIFICATION_EXPIRED,
    CLARIFICATION_ROUND_UPDATE,
}
"""Complete set of clarification-related WebSocket event types (Req 12.1)."""

# Channel used for clarification broadcasts
UPLOADS_CHANNEL = "uploads"


class WebSocketManager:
    """Manages WebSocket connections grouped by channel and broadcasts events.

    The manager supports any arbitrary event type via the generic `broadcast()`
    method. Clarification-specific events are additionally supported via
    `broadcast_clarification()` which enforces that submission_id and session_id
    are always present in the payload (Req 12.4).
    """

    def __init__(self) -> None:
        self.channels: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        self.channels[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        self.channels[channel].discard(websocket)

    async def broadcast(self, channel: str, event: str, payload: dict) -> None:
        """Broadcast an event with payload to all connected clients on a channel.

        Args:
            channel: The channel name (e.g., "uploads").
            event: The event type string (e.g., "clarification_required").
            payload: The event payload dict. For clarification events, must
                     include submission_id and session_id per Req 12.4.
        """
        message = json.dumps({"event": event, "payload": payload}, default=str)
        stale: list[WebSocket] = []
        for websocket in self.channels[channel]:
            try:
                await websocket.send_text(message)
            except RuntimeError:
                stale.append(websocket)
        for websocket in stale:
            self.channels[channel].discard(websocket)

    async def broadcast_clarification(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Broadcast a clarification event on the 'uploads' channel.

        This is a convenience method that validates the event type is one of
        the known clarification events and ensures the required fields
        (submission_id, session_id) are present in the payload.

        Args:
            event_type: One of CLARIFICATION_REQUIRED, CLARIFICATION_RESOLVED,
                        CLARIFICATION_EXPIRED, CLARIFICATION_ROUND_UPDATE.
            payload: The event payload dict. Must include submission_id and session_id.

        Raises:
            ValueError: If event_type is not a valid clarification event type.
            ValueError: If payload is missing submission_id or session_id.

        Requirements: 12.1, 12.4
        """
        if event_type not in CLARIFICATION_EVENT_TYPES:
            raise ValueError(
                f"Unknown clarification event type: {event_type!r}. "
                f"Valid types: {CLARIFICATION_EVENT_TYPES}"
            )

        # Enforce required fields per Req 12.4
        missing_fields = []
        if "submission_id" not in payload:
            missing_fields.append("submission_id")
        if "session_id" not in payload:
            missing_fields.append("session_id")

        if missing_fields:
            raise ValueError(
                f"Clarification event payload missing required fields: {missing_fields}. "
                f"All clarification events must include submission_id and session_id (Req 12.4)."
            )

        await self.broadcast(UPLOADS_CHANNEL, event_type, payload)


ws_manager = WebSocketManager()
