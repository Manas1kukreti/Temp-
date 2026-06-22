import { useCallback, useState } from "react";
import { useWebSocket } from "./useWebSocket.js";

/**
 * useClarificationSocket — Custom hook that subscribes to the "uploads" WebSocket
 * channel and filters for clarification lifecycle events. Exposes clarification
 * session state that a parent component can use to conditionally render
 * ClarificationPanel.
 *
 * Handles:
 *   - "clarification_required": activates session with received questions
 *   - "clarification_round_update": updates panel with new questions, token, round count
 *   - "clarification_resolved": dismisses panel, signals success
 *   - "clarification_expired": dismisses panel, signals expiration
 *
 * @param {string|null} submissionId — Optional filter to only react to events for a specific submission.
 * @returns {object} Clarification state and control methods.
 *
 * Validates: Requirements 3.1, 12.1, 12.3
 */
export function useClarificationSocket(submissionId = null) {
  const [session, setSession] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | active | resolved | expired

  const handleMessage = useCallback(
    (message) => {
      const { event, payload } = message || {};
      if (!event || !payload) return;

      // If a submissionId filter is provided, ignore events for other submissions
      if (submissionId && String(payload.submission_id) !== String(submissionId)) {
        return;
      }

      switch (event) {
        case "clarification_required":
          setSession({
            sessionId: payload.session_id,
            submissionId: payload.submission_id,
            intentVersion: payload.intent_version,
            revisionToken: payload.revision_token,
            expiresAt: payload.expires_at,
            questions: payload.questions || [],
            roundCount: 0,
            maxRounds: payload.max_rounds || 2,
          });
          setStatus("active");
          break;

        case "clarification_round_update":
          setSession((prev) => {
            if (!prev) return prev;
            // Only update if this event matches the current session
            if (payload.session_id && prev.sessionId !== payload.session_id) {
              return prev;
            }
            return {
              ...prev,
              questions: payload.questions || prev.questions,
              revisionToken: payload.revision_token || prev.revisionToken,
              roundCount: payload.round_count ?? prev.roundCount,
              intentVersion: payload.intent_version ?? prev.intentVersion,
            };
          });
          break;

        case "clarification_resolved":
          setSession((prev) => {
            if (!prev) return null;
            if (payload.session_id && prev.sessionId !== payload.session_id) {
              return prev;
            }
            return {
              ...prev,
              intentVersion: payload.intent_version ?? prev.intentVersion,
            };
          });
          setStatus("resolved");
          break;

        case "clarification_expired":
          setSession((prev) => {
            if (!prev) return null;
            if (payload.session_id && prev.sessionId !== payload.session_id) {
              return prev;
            }
            return prev;
          });
          setStatus("expired");
          break;

        default:
          // Not a clarification event — ignore
          break;
      }
    },
    [submissionId],
  );

  useWebSocket("uploads", handleMessage);

  /**
   * Manually refresh session state — useful after a 409 (stale token)
   * response where the error payload contains updated session info.
   */
  const refreshSession = useCallback((newState) => {
    setSession((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        questions: newState.questions ?? prev.questions,
        roundCount: newState.roundCount ?? newState.round_count ?? prev.roundCount,
        revisionToken: newState.revisionToken ?? newState.revision_token ?? prev.revisionToken,
        intentVersion: newState.intentVersion ?? newState.intent_version ?? prev.intentVersion,
      };
    });
  }, []);

  /**
   * Dismiss the clarification panel manually (e.g., after navigating away).
   */
  const dismiss = useCallback(() => {
    setSession(null);
    setStatus("idle");
  }, []);

  return {
    /** Current clarification session data (null when no active session) */
    session,
    /** Session lifecycle status: "idle" | "active" | "resolved" | "expired" */
    status,
    /** Whether clarification panel should be visible */
    isActive: status === "active" && session !== null,
    /** Whether the session resolved successfully */
    isResolved: status === "resolved",
    /** Whether the session expired */
    isExpired: status === "expired",
    /** Manually update session state (e.g., from 409 error response) */
    refreshSession,
    /** Dismiss the panel and reset state */
    dismiss,
  };
}
