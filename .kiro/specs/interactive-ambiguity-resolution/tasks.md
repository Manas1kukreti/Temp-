# Implementation Plan: Interactive Ambiguity Resolution

## Overview

This plan implements the interactive ambiguity resolution subsystem for FinFlow. The system detects resolvable ambiguities in user prompts, creates clarification sessions, generates structured questions, delivers them via WebSocket, and patches intents with user answers — all while preserving prior grounding work through selective reprocessing.

Implementation proceeds in layers: data models and enums first, then core backend services (session management, question generation, validation, patching), followed by the API endpoint, WebSocket integration, expiration scheduler, and finally the frontend ClarificationPanel component.

## Tasks

- [x] 1. Set up data models, enums, and database schema
  - [x] 1.1 Create ClarificationOutcome enum and data model classes
    - Create `backend/app/models/clarification.py` with:
      - `ClarificationOutcome` enum (RESOLVED, STILL_AMBIGUOUS, INVALID_RESPONSE, CONFLICT_INTRODUCED, MAX_ROUNDS_EXCEEDED, SESSION_EXPIRED)
      - `ClarificationSession` SQLAlchemy model (id, submission_id, intent_version, round_count, max_rounds, status, revision_token, expires_at, created_at, updated_at, outcome)
      - `ClarificationQuestion` SQLAlchemy model (id, session_id, round_number, intent_path, reason_code, question_text, candidate_options, free_text_enabled, user_answer)
      - `CanonicalIntentRevision` SQLAlchemy model (id, intent_id, intent_revision, intent_hash, parent_intent_id, payload, created_at)
    - Add `ReasonCode` enum (MULTIPLE_COLUMN_MATCHES, AMBIGUOUS_REFERENCE, LOW_CONFIDENCE_SCORE, MISSING_COLUMN, CONFLICTING_EVIDENCE)
    - Add `SessionStatus` enum (active, resolved, expired, max_rounds_exceeded)
    - _Requirements: 1.4, 2.2, 8.1, 8.2, 14.1, 14.2_

  - [x] 1.2 Create Pydantic API schemas for clarification payloads
    - Create `backend/app/schemas/clarification.py` with:
      - `ClarificationResponsePayload` (session_id, intent_version, revision_token optional, answers list)
      - `ClarificationAnswer` (question_id, selected_option optional, free_text optional)
      - `ClarificationOutcomeResponse` (outcome, submission_id, session_id, intent_version, error_details optional, remaining_questions optional)
      - `QuestionError` (question_id, reason)
      - `ClarificationQuestionSchema` for serialization
    - _Requirements: 4.1, 4.2, 5.4_

  - [x] 1.3 Add "awaiting_clarification" to SubmissionStatus enum
    - Update the existing `SubmissionStatus` enum in the submissions model to include `awaiting_clarification`
    - Add `intent_revision` and `intent_hash` fields to the Submission model
    - _Requirements: 1.3, 8.3, 13.1_

  - [x] 1.4 Create database migration for clarification tables
    - Create Alembic migration adding `clarification_sessions`, `clarification_questions`, and `canonical_intent_revisions` tables
    - Add index on `clarification_sessions.submission_id` and `clarification_sessions.status`
    - Add index on `clarification_sessions.expires_at` for expiration queries
    - _Requirements: 1.4, 14.4_

- [x] 2. Implement core backend services
  - [x] 2.1 Implement SessionRepository
    - Create `backend/app/services/clarification_session_repo.py` with:
      - `create(session)` — persist new session
      - `get_active_by_submission(submission_id)` — fetch active session for a submission
      - `update(session)` — update session state
      - `get_expired_sessions()` — query sessions past expires_at with status=active
      - `record_round(session_id, questions, answers, outcome)` — persist question-answer history per round
    - Generate cryptographically random revision_token (secrets.token_urlsafe, at least 32 chars)
    - _Requirements: 1.4, 9.1, 9.5, 14.4_

  - [x] 2.2 Implement QuestionGenerator
    - Create `backend/app/services/clarification_questions.py` with:
      - `generate(unresolved_fields, intent_package, intent)` — produce one ClarificationQuestion per unresolved field
      - For MULTIPLE_COLUMN_MATCHES: candidate_options from top-scoring columns ordered by confidence descending, question_text presents options only
      - For AMBIGUOUS_REFERENCE: question_text describes ambiguity with competing interpretations as options
      - For LOW_CONFIDENCE_SCORE: "Which column did you mean by '{raw_reference}'?" with top candidates
      - For MISSING_COLUMN: prompt for free text with column name
      - For CONFLICTING_EVIDENCE: describe conflict and ask for disambiguation
      - Always append "none_of_these" as final candidate_option
      - Set free_text_enabled = True on all questions
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 14.1, 14.2, 14.3_

  - [x]* 2.3 Write property test for QuestionGenerator structure (Property 2)
    - **Property 2: Question generation produces correct structure**
    - Verify: exactly one question per unresolved field, valid question_id, non-empty question_text, correct intent_path, valid reason_code, "none_of_these" as final option, free_text_enabled=True
    - **Validates: Requirements 2.1, 2.2, 2.3, 14.1, 14.2**

  - [x]* 2.4 Write property test for candidate ordering (Property 3)
    - **Property 3: Candidate options ordered by confidence**
    - Verify: for MULTIPLE_COLUMN_MATCHES reason_code, candidate_options (excluding "none_of_these") are ordered by confidence descending
    - **Validates: Requirements 2.4**

  - [x] 2.5 Implement ResponseValidator
    - Create `backend/app/services/clarification_validator.py` with:
      - `validate(session, response)` returning ValidationResult
      - Check revision_token matches (if provided; skip if None per Req 9.4)
      - Check intent_version matches session's current version
      - Check all question_ids belong to the current session
      - Check selected_option is in the question's candidate_options or is "none_of_these"
      - Check "none_of_these" answers have non-empty free_text
      - Return per-question error_details for failures
    - _Requirements: 4.3, 4.4, 4.5, 5.1, 5.2, 5.4, 9.2, 9.3, 9.4_

  - [x]* 2.6 Write property test for stale token/version rejection (Property 4)
    - **Property 4: Stale token and version rejection**
    - Verify: mismatched revision_token → 409 + SESSION_EXPIRED; mismatched intent_version → 409 + stale-version; no modification to intent or round_count
    - **Validates: Requirements 4.3, 4.4, 9.2, 9.3**

  - [x]* 2.7 Write property test for invalid response rejection (Property 5)
    - **Property 5: Invalid response rejection without round penalty**
    - Verify: invalid selected_option or "none_of_these" with empty free_text returns INVALID_RESPONSE with error_details; round_count unchanged
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

  - [x]* 2.8 Write property test for optional token bypass (Property 12)
    - **Property 12: Optional token bypass**
    - Verify: when revision_token is None/absent, submission proceeds without token verification failure
    - **Validates: Requirements 9.4**

  - [x] 2.9 Implement IntentPatcher
    - Create `backend/app/services/intent_patcher.py` with:
      - `apply_patch(intent, intent_package, answers)` → PatchResult
      - For selected candidate_option: set resolved_column to selection, confidence=1.0, resolution_method="user_clarification"
      - For "none_of_these" + free_text: set raw_reference to free_text, re-run column resolution
      - Increment intent_revision, compute new intent_hash (SHA-256), set parent_intent_id
      - Persist CanonicalIntentRevision with previous version's full state
      - Re-run grounding and validation (skip extraction/normalization)
      - Detect conflicts and set conflict flag in PatchResult
      - Return new intent, remaining unresolved fields list, and conflict flag
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.1, 8.2, 8.3_

  - [x]* 2.10 Write property test for selective patching (Property 6)
    - **Property 6: Selective patching preserves untargeted fields**
    - Verify: fields not targeted by answers remain byte-for-byte identical after patch
    - **Validates: Requirements 6.1**

  - [x]* 2.11 Write property test for resolution correctness (Property 7)
    - **Property 7: Resolution correctness on patch**
    - Verify: selected candidate → resolved_column with confidence 1.0 + resolution_method "user_clarification"; "none_of_these" + free_text → raw_reference updated
    - **Validates: Requirements 6.2, 6.3**

  - [x]* 2.12 Write property test for intent versioning (Property 8)
    - **Property 8: Intent versioning invariant**
    - Verify: new revision = previous + 1, hash differs, parent_intent_id correct, CanonicalIntentRevision persisted
    - **Validates: Requirements 6.4, 8.1, 8.2, 8.3**

- [x] 3. Implement ClarificationService orchestrator
  - [x] 3.1 Implement ClarificationService.initiate_session
    - Create `backend/app/services/clarification_service.py` with:
      - `initiate_session(submission_id, intent, intent_package, ambiguity_score)`:
        - Check ambiguity_score against threshold; if below, quarantine directly and return
        - Create ClarificationSession with round_count=0, generate revision_token
        - Set session expires_at to now + configurable timeout (default 30 min)
        - Call QuestionGenerator.generate() for unresolved fields
        - Persist session and questions via SessionRepository
        - Transition submission status to "awaiting_clarification"
        - Broadcast "clarification_required" WebSocket event with full payload
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 10.1_

  - [x]* 3.2 Write property test for ambiguity routing (Property 1)
    - **Property 1: Ambiguity routing partitions correctly**
    - Verify: score >= threshold → session created + status "awaiting_clarification"; score < threshold → quarantined without session
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 3.3 Implement ClarificationService.handle_response
    - In `clarification_service.py` add:
      - `handle_response(submission_id, response)`:
        - Load active session, validate via ResponseValidator
        - If validation fails with token/version mismatch: return 409 with appropriate outcome
        - If validation fails with invalid answers: return INVALID_RESPONSE (no round increment)
        - If valid: increment round_count, call IntentPatcher.apply_patch()
        - After patch, check PatchResult:
          - conflict_flag True → return CONFLICT_INTRODUCED (no round increment)
          - no remaining unresolved fields → RESOLVED, dispatch to execution
          - remaining fields + round_count < 2 → STILL_AMBIGUOUS, generate new questions, new revision_token
          - remaining fields + round_count == 2 → MAX_ROUNDS_EXCEEDED, quarantine
        - Record round history via SessionRepository
        - Broadcast appropriate WebSocket event based on outcome
    - _Requirements: 5.3, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 8.4, 12.2, 12.3_

  - [x]* 3.4 Write property test for round management (Property 9)
    - **Property 9: Round management enforces limits**
    - Verify: valid submissions increment round_count by 1; INVALID_RESPONSE does not increment; round_count never exceeds 2
    - **Validates: Requirements 5.3, 7.1**

  - [x]* 3.5 Write property test for outcome routing (Property 10)
    - **Property 10: Outcome routing after re-grounding**
    - Verify: no unresolved → RESOLVED; unresolved + rounds < 2 → STILL_AMBIGUOUS + new questions; unresolved + rounds == 2 → MAX_ROUNDS_EXCEEDED + quarantine; conflict → CONFLICT_INTRODUCED
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5**

  - [x] 3.6 Implement ClarificationService.expire_session
    - In `clarification_service.py` add:
      - `expire_session(session_id)`:
        - Transition session status to "expired", set outcome to SESSION_EXPIRED
        - Quarantine submission with reason "clarification_session_expired"
        - Broadcast "clarification_expired" WebSocket event
    - _Requirements: 10.2, 10.3_

- [x] 4. Checkpoint - Ensure all core service tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement API endpoint and expiration scheduler
  - [x] 5.1 Implement ClarificationRouter API endpoint
    - Create `backend/app/api/clarification.py` with:
      - POST `/{submission_id}/clarify` endpoint
      - Validate submission exists and has active clarification session
      - Check session not expired (return 410 Gone if expired)
      - Delegate to ClarificationService.handle_response()
      - Map ClarificationOutcome to appropriate HTTP status codes:
        - INVALID_RESPONSE → 200 with error_details
        - SESSION_EXPIRED → 410 Gone
        - Stale token/version → 409 Conflict
        - Invalid question_id → 400 Bad Request
        - RESOLVED, STILL_AMBIGUOUS, MAX_ROUNDS_EXCEEDED, CONFLICT_INTRODUCED → 200
      - Register router in app with prefix `/api/uploads`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.3_

  - [x] 5.2 Implement ExpirationScheduler
    - Create `backend/app/services/clarification_expiration.py` with:
      - `run_cleanup()` — periodic task that:
        - Queries all sessions where expires_at < now and status = active
        - For each: call ClarificationService.expire_session()
        - Return count of expired sessions processed
      - Register as background task (e.g., via arq worker or asyncio periodic task)
    - _Requirements: 10.1, 10.2, 10.4_

  - [x]* 5.3 Write property test for session expiration enforcement (Property 13)
    - **Property 13: Session expiration enforcement**
    - Verify: expired session → response returns 410 + SESSION_EXPIRED; cleanup task quarantines submission with correct reason
    - **Validates: Requirements 10.2, 10.3**

  - [x]* 5.4 Write property test for revision token generation (Property 11)
    - **Property 11: Revision token generation**
    - Verify: tokens are at least 32 characters, cryptographically random, unique across rounds
    - **Validates: Requirements 9.1, 9.5**

- [x] 6. Implement WebSocket event integration
  - [x] 6.1 Add clarification WebSocket event types and broadcast logic
    - Update `backend/app/services/websocket_manager.py` (or create helper) to:
      - Support event types: "clarification_required", "clarification_resolved", "clarification_expired", "clarification_round_update"
      - Broadcast on "uploads" channel with correct payloads per design spec
      - Include submission_id and session_id in every clarification event
      - "clarification_required": session_id, submission_id, intent_version, revision_token, expires_at, questions
      - "clarification_resolved": session_id, submission_id, intent_version, outcome
      - "clarification_expired": session_id, submission_id
      - "clarification_round_update": session_id, submission_id, round_count, revision_token, questions
    - _Requirements: 3.1, 3.2, 3.3, 12.1, 12.2, 12.3, 12.4_

  - [x]* 6.2 Write property test for WebSocket event payload correctness (Property 14)
    - **Property 14: WebSocket event payload correctness**
    - Verify: all events include submission_id + session_id; RESOLVED → only "clarification_resolved" (no round_update); STILL_AMBIGUOUS → "clarification_round_update" with questions, token, round_count
    - **Validates: Requirements 12.2, 12.3, 12.4**

- [x] 7. Checkpoint - Ensure all backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement frontend ClarificationPanel and status integration
  - [x] 8.1 Create ClarificationPanel React component
    - Create `frontend/src/components/ClarificationPanel.tsx` with:
      - Props: submissionId, sessionId, questions, roundCount, maxRounds, expiresAt, revisionToken, intentVersion
      - Render all questions with candidate_options as radio button groups
      - "None of these" option reveals free-text input with placeholder "Describe what you meant"
      - Display round count ("Round 1 of 2") and session expiration countdown timer
      - Submit button calls POST `/api/uploads/{submission_id}/clarify`
      - Handle INVALID_RESPONSE: highlight invalid answers with per-question error messages, allow resubmission
      - Handle RESOLVED: show success message, transition to running state
      - Handle SESSION_EXPIRED: display "Session expired" message
      - Handle stale token (409): auto-refresh session state from error response
      - Full keyboard accessibility with ARIA labels on all interactive elements
      - Radio groups with proper `role="radiogroup"` and `aria-label`
      - Free-text inputs with `aria-describedby` for error messages
      - Disable free-text input when validation errors exist on the question
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [x] 8.2 Integrate WebSocket events with ClarificationPanel
    - Wire the existing useWebSocket hook to listen for clarification events:
      - "clarification_required": render ClarificationPanel with received questions
      - "clarification_round_update": update panel with new questions, token, round count
      - "clarification_resolved": dismiss panel, show success
      - "clarification_expired": dismiss panel, show expiration message
    - _Requirements: 3.1, 12.1, 12.3_

  - [x] 8.3 Update SubmissionsPage for "awaiting_clarification" status
    - Update `mapWorkflowStatus` function to map "awaiting_clarification" to "clarification" display status with validation fallback
    - Add distinct visual indicator and "Needs your input" label for submissions in this status
    - Show ClarificationPanel in job detail view when status is "awaiting_clarification"
    - _Requirements: 13.2, 13.3, 13.4_

  - [x] 8.4 Write frontend unit tests for ClarificationPanel
    - Test: renders questions with selectable options
    - Test: "None of these" reveals free-text input
    - Test: form submission calls correct API endpoint
    - Test: error state highlighting on INVALID_RESPONSE
    - Test: success state transition on RESOLVED
    - Test: keyboard navigation and ARIA compliance
    - Test: countdown timer updates
    - Test: mapWorkflowStatus maps "awaiting_clarification" correctly
    - _Requirements: 11.1, 11.2, 11.4, 11.5, 11.6, 11.7, 13.3_

- [x] 9. Integration wiring and end-to-end flow
  - [x] 9.1 Integrate ClarificationService into Semantic Pipeline
    - Modify `backend/app/services/semantic_pipeline.py` to:
      - After resolution step, check if resolution_status == "needs_clarification"
      - Call ClarificationService.initiate_session() with ambiguity_score and unresolved fields
      - If session is created, halt further pipeline processing and return early
      - If session not created (below threshold), proceed with existing quarantine logic
    - _Requirements: 1.1, 1.2_

  - [x] 9.2 Add REST polling endpoint for clarification status
    - Create GET `/api/uploads/{submission_id}/clarification-status` endpoint
      - Return current session state (questions, round, expiration) for frontend resync after WebSocket disconnect
    - _Requirements: Error handling graceful degradation_

  - [ ] 9.3 Write integration tests for full clarification flow
    - Test POST `/api/uploads/{id}/clarify` endpoint contract with httpx AsyncClient
    - Test multi-round clarification: round 1 → STILL_AMBIGUOUS → round 2 → RESOLVED
    - Test expiration cleanup task execution
    - Test WebSocket event delivery for all event types
    - _Requirements: 4.1, 7.2, 7.3, 10.4, 12.1_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- The backend uses Python (FastAPI + SQLAlchemy + Hypothesis for PBT)
- The frontend uses TypeScript (React + Vitest + React Testing Library)
- All 14 correctness properties from the design are covered by property test sub-tasks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4"] },
    { "id": 2, "tasks": ["2.1", "2.2"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.5"] },
    { "id": 4, "tasks": ["2.6", "2.7", "2.8", "2.9"] },
    { "id": 5, "tasks": ["2.10", "2.11", "2.12", "3.1"] },
    { "id": 6, "tasks": ["3.2", "3.3"] },
    { "id": 7, "tasks": ["3.4", "3.5", "3.6"] },
    { "id": 8, "tasks": ["5.1", "5.2"] },
    { "id": 9, "tasks": ["5.3", "5.4", "6.1"] },
    { "id": 10, "tasks": ["6.2", "8.1"] },
    { "id": 11, "tasks": ["8.2", "8.3"] },
    { "id": 12, "tasks": ["8.4", "9.1", "9.2"] },
    { "id": 13, "tasks": ["9.3"] }
  ]
}
```
