import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client.js";

/**
 * ClarificationPanel — Interactive UI for resolving ambiguous intent fields.
 *
 * Props:
 *   submissionId: string
 *   sessionId: string
 *   questions: ClarificationQuestion[]
 *   roundCount: number
 *   maxRounds: number
 *   expiresAt: string (ISO timestamp)
 *   revisionToken: string
 *   intentVersion: number
 *   onResolved?: () => void
 *   onSessionExpired?: () => void
 *   onSessionRefresh?: (newState) => void
 */

function computeTimeRemaining(expiresAt) {
  const diff = new Date(expiresAt).getTime() - Date.now();
  if (diff <= 0) return { expired: true, minutes: 0, seconds: 0, display: "Expired" };
  const minutes = Math.floor(diff / 60000);
  const seconds = Math.floor((diff % 60000) / 1000);
  return {
    expired: false,
    minutes,
    seconds,
    display: `${minutes}m ${String(seconds).padStart(2, "0")}s remaining`,
  };
}

export default function ClarificationPanel({
  submissionId,
  sessionId,
  questions,
  roundCount,
  maxRounds,
  expiresAt,
  revisionToken,
  intentVersion,
  onResolved,
  onSessionExpired,
  onSessionRefresh,
}) {
  const [answers, setAnswers] = useState(() =>
    Object.fromEntries(questions.map((q) => [q.question_id, { selected: null, freeText: "" }]))
  );
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(() => computeTimeRemaining(expiresAt));
  const timerRef = useRef(null);

  // Reset answers when questions change (e.g., new round)
  useEffect(() => {
    setAnswers(
      Object.fromEntries(questions.map((q) => [q.question_id, { selected: null, freeText: "" }]))
    );
    setErrors({});
    setOutcome(null);
  }, [questions]);

  // Countdown timer
  useEffect(() => {
    const tick = () => {
      const remaining = computeTimeRemaining(expiresAt);
      setTimeRemaining(remaining);
      if (remaining.expired) {
        clearInterval(timerRef.current);
        setOutcome("SESSION_EXPIRED");
        onSessionExpired?.();
      }
    };
    tick();
    timerRef.current = setInterval(tick, 1000);
    return () => clearInterval(timerRef.current);
  }, [expiresAt, onSessionExpired]);

  const handleOptionSelect = useCallback((questionId, option) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: { ...prev[questionId], selected: option, freeText: option === "none_of_these" ? prev[questionId]?.freeText || "" : "" },
    }));
    // Clear error for this question when user makes a new selection
    setErrors((prev) => {
      const next = { ...prev };
      delete next[questionId];
      return next;
    });
  }, []);

  const handleFreeTextChange = useCallback((questionId, value) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: { ...prev[questionId], freeText: value },
    }));
  }, []);

  const buildPayload = useCallback(() => {
    const answersList = questions.map((q) => {
      const answer = answers[q.question_id];
      if (answer?.selected === "none_of_these") {
        return { question_id: q.question_id, selected_option: "none_of_these", free_text: answer.freeText };
      }
      return { question_id: q.question_id, selected_option: answer?.selected || null, free_text: null };
    });
    return {
      session_id: sessionId,
      intent_version: intentVersion,
      revision_token: revisionToken,
      answers: answersList,
    };
  }, [questions, answers, sessionId, intentVersion, revisionToken]);

  const handleSubmit = useCallback(async (event) => {
    event.preventDefault();
    if (submitting) return;

    setSubmitting(true);
    setErrors({});
    setOutcome(null);

    try {
      const response = await api.post(`/uploads/${submissionId}/clarify`, buildPayload());
      const data = response.data;

      switch (data.outcome) {
        case "INVALID_RESPONSE":
          setOutcome("INVALID_RESPONSE");
          if (data.error_details) {
            const errorMap = {};
            data.error_details.forEach((err) => {
              errorMap[err.question_id] = err.reason;
            });
            setErrors(errorMap);
          }
          break;
        case "RESOLVED":
          setOutcome("RESOLVED");
          onResolved?.();
          break;
        case "SESSION_EXPIRED":
          setOutcome("SESSION_EXPIRED");
          onSessionExpired?.();
          break;
        case "STILL_AMBIGUOUS":
          // New round will be delivered via WebSocket or via response payload
          if (data.remaining_questions) {
            onSessionRefresh?.({
              questions: data.remaining_questions,
              roundCount: data.round_count || roundCount + 1,
              revisionToken: data.revision_token || revisionToken,
              intentVersion: data.intent_version || intentVersion,
            });
          }
          break;
        case "MAX_ROUNDS_EXCEEDED":
          setOutcome("MAX_ROUNDS_EXCEEDED");
          break;
        case "CONFLICT_INTRODUCED":
          setOutcome("CONFLICT_INTRODUCED");
          break;
        default:
          break;
      }
    } catch (err) {
      const status = err.response?.status;
      if (status === 409) {
        // Stale token — auto-refresh session state from error response
        const errorData = err.response?.data;
        if (errorData) {
          onSessionRefresh?.(errorData);
        }
        setOutcome("STALE_TOKEN");
      } else if (status === 410) {
        setOutcome("SESSION_EXPIRED");
        onSessionExpired?.();
      } else {
        setOutcome("ERROR");
      }
    } finally {
      setSubmitting(false);
    }
  }, [submitting, submissionId, buildPayload, onResolved, onSessionExpired, onSessionRefresh, roundCount, revisionToken, intentVersion]);

  // Resolved state
  if (outcome === "RESOLVED") {
    return (
      <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Clarification resolved">
        <div className="ff-clarification-success">
          <h3>Ambiguities resolved</h3>
          <p className="ff-copy-muted">Your answers have been applied. The job is now running.</p>
        </div>
      </section>
    );
  }

  // Session expired state
  if (outcome === "SESSION_EXPIRED") {
    return (
      <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Session expired">
        <div className="ff-clarification-expired">
          <h3>Session expired</h3>
          <p className="ff-copy-muted">
            The clarification session has expired. This job has been moved to quarantine.
          </p>
        </div>
      </section>
    );
  }

  // Stale token state — will be refreshed by parent
  if (outcome === "STALE_TOKEN") {
    return (
      <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Refreshing session">
        <div className="ff-clarification-refresh">
          <h3>Session updated</h3>
          <p className="ff-copy-muted">
            The session state has been updated. Refreshing questions...
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      className="ff-panel ff-panel--dense ff-clarification-panel"
      aria-label="Clarification questions"
    >
      <div className="ff-clarification-panel__header">
        <h3>Clarification needed</h3>
        <div className="ff-clarification-panel__meta">
          <span className="ff-clarification-panel__round" aria-label={`Round ${roundCount + 1} of ${maxRounds}`}>
            Round {roundCount + 1} of {maxRounds}
          </span>
          <span
            className={`ff-clarification-panel__timer ${timeRemaining.expired ? "ff-clarification-panel__timer--expired" : ""}`}
            aria-live="polite"
            aria-label={`Session expires in ${timeRemaining.display}`}
          >
            {timeRemaining.display}
          </span>
        </div>
      </div>

      {outcome === "INVALID_RESPONSE" && (
        <div className="ff-clarification-panel__alert" role="alert" aria-label="Validation errors">
          <p>Some answers need correction. Please fix the highlighted questions and resubmit.</p>
        </div>
      )}

      {outcome === "MAX_ROUNDS_EXCEEDED" && (
        <div className="ff-clarification-panel__alert ff-clarification-panel__alert--warning" role="alert">
          <p>Maximum clarification rounds reached. The job has been quarantined for review.</p>
        </div>
      )}

      {outcome === "CONFLICT_INTRODUCED" && (
        <div className="ff-clarification-panel__alert ff-clarification-panel__alert--warning" role="alert">
          <p>Your answers introduced a conflict. Please review and adjust your selections.</p>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        <div className="ff-clarification-panel__questions">
          {questions.map((question, index) => {
            const questionError = errors[question.question_id];
            const currentAnswer = answers[question.question_id];
            const hasError = Boolean(questionError);
            const errorId = `error-${question.question_id}`;

            return (
              <fieldset
                key={question.question_id}
                className={`ff-clarification-question ${hasError ? "ff-clarification-question--error" : ""}`}
                aria-label={`Question ${index + 1}: ${question.question_text}`}
              >
                <legend className="ff-clarification-question__text">
                  {question.question_text}
                </legend>

                <div
                  role="radiogroup"
                  aria-label={`Options for: ${question.question_text}`}
                  className="ff-clarification-question__options"
                >
                  {question.candidate_options
                    .filter((opt) => opt !== "none_of_these")
                    .map((option) => (
                      <label
                        key={option}
                        className={`ff-clarification-option ${currentAnswer?.selected === option ? "ff-clarification-option--selected" : ""}`}
                      >
                        <input
                          type="radio"
                          name={`question-${question.question_id}`}
                          value={option}
                          checked={currentAnswer?.selected === option}
                          onChange={() => handleOptionSelect(question.question_id, option)}
                          aria-label={option}
                        />
                        <span className="ff-clarification-option__label">{option}</span>
                      </label>
                    ))}

                  {/* None of these option */}
                  <label
                    className={`ff-clarification-option ff-clarification-option--none ${currentAnswer?.selected === "none_of_these" ? "ff-clarification-option--selected" : ""}`}
                  >
                    <input
                      type="radio"
                      name={`question-${question.question_id}`}
                      value="none_of_these"
                      checked={currentAnswer?.selected === "none_of_these"}
                      onChange={() => handleOptionSelect(question.question_id, "none_of_these")}
                      aria-label="None of these"
                    />
                    <span className="ff-clarification-option__label">None of these</span>
                  </label>
                </div>

                {/* Free-text input revealed when "None of these" is selected */}
                {currentAnswer?.selected === "none_of_these" && (
                  <div className="ff-clarification-question__freetext">
                    <input
                      type="text"
                      value={currentAnswer.freeText}
                      onChange={(e) => handleFreeTextChange(question.question_id, e.target.value)}
                      placeholder="Describe what you meant"
                      disabled={hasError}
                      aria-label={`Free text input for question: ${question.question_text}`}
                      aria-describedby={hasError ? errorId : undefined}
                      className="ff-clarification-question__freetext-input"
                    />
                  </div>
                )}

                {/* Per-question error message */}
                {hasError && (
                  <p
                    id={errorId}
                    className="ff-clarification-question__error"
                    role="alert"
                    aria-live="assertive"
                  >
                    {questionError}
                  </p>
                )}
              </fieldset>
            );
          })}
        </div>

        <div className="ff-clarification-panel__actions">
          <button
            type="submit"
            className="ff-primary-button"
            disabled={submitting || timeRemaining.expired}
            aria-label="Submit clarification answers"
          >
            {submitting ? "Submitting..." : "Submit answers"}
          </button>
        </div>
      </form>
    </section>
  );
}
