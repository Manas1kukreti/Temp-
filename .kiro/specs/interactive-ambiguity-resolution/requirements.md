# Requirements Document

## Introduction

Interactive ambiguity resolution enables the FinFlow system to recover from ambiguous user prompts without quarantining jobs or requiring re-prompting. When the semantic pipeline detects resolvable ambiguities (unresolved column references, unclear projections, ambiguous filter targets), the system creates a clarification session, generates structured questions targeting the specific unresolved fields, delivers them to the user via WebSocket, and patches the existing CanonicalIntent with the user's answers. This avoids full intent regeneration and preserves prior grounding work through selective reprocessing.

## Glossary

- **Clarification_Session**: A stateful server-side object that tracks an interactive clarification exchange between the system and the user for a specific submission. Contains session_id, intent_version, round count, question history, and expiration timestamp.
- **Clarification_Question**: A structured question targeting a specific unresolved intent path, containing question_id, the intent_path it resolves, a reason_code, candidate options, and a free-text input field.
- **Clarification_Response**: A user-submitted answer to one or more Clarification_Questions, including session_id, intent_version, question_id, revision_token, and selected values or free-text input.
- **Clarification_Round**: One complete cycle of sending questions and receiving answers. A maximum of 2 semantic clarification rounds are permitted per session.
- **Clarification_Outcome**: The result of processing a Clarification_Response. One of: RESOLVED, STILL_AMBIGUOUS, INVALID_RESPONSE, CONFLICT_INTRODUCED, MAX_ROUNDS_EXCEEDED, SESSION_EXPIRED.
- **Semantic_Pipeline**: The backend service (semantic_pipeline.py) that orchestrates extraction, normalization, grounding, coverage, repair, and compilation of user prompts into CanonicalIntent actions.
- **CanonicalIntent**: The versioned, structured representation of a user's data-processing instructions, containing actions, column references, filter conditions, and output format.
- **Intent_Patch**: A targeted modification to specific fields within an existing CanonicalIntent, as opposed to full regeneration from the original prompt.
- **Intent_Version**: A monotonically increasing revision number tracking successive patches to a CanonicalIntent (v1 → clarification → v2).
- **Unresolved_Field**: A specific path within the CanonicalIntent (e.g., "actions[1].conditions[0].field") where the grounding or resolution step could not determine a definitive column match.
- **Reason_Code**: A classification of why a field is unresolved. Values include MULTIPLE_COLUMN_MATCHES, AMBIGUOUS_REFERENCE, LOW_CONFIDENCE_SCORE, MISSING_COLUMN, CONFLICTING_EVIDENCE.
- **Revision_Token**: A server-generated opaque token tied to a specific intent version that prevents stale responses from being applied to a newer version.
- **Ambiguity_Score**: A numeric score derived from the semantic pipeline indicating how resolvable an ambiguity is. Scores below a system-defined threshold trigger direct quarantine rather than clarification.
- **WebSocket_Manager**: The existing backend service (websocket_manager.py) that manages channel-based WebSocket connections and broadcasts events to connected frontends.
- **Submission**: The database entity representing a user-uploaded file and associated processing instruction.
- **IntentPackage**: The shared versioned schema-resolution artifact containing resolved columns, unresolved fields, grounding results, and the patch_column() method.

## Requirements

### Requirement 1: Ambiguity Detection and Routing

**User Story:** As a user, I want the system to detect when my prompt has resolvable ambiguities and route them to interactive clarification rather than quarantine, so that I can resolve issues without losing my job context.

#### Acceptance Criteria

1. WHEN the Semantic_Pipeline returns a resolution_status of "needs_clarification" and the Ambiguity_Score is at or above the clarification threshold, THE Clarification_Session SHALL be created for the associated Submission.
2. WHEN the Semantic_Pipeline returns a resolution_status of "needs_clarification" and the Ambiguity_Score is below the clarification threshold, THE Submission SHALL be quarantined directly without creating a Clarification_Session.
3. WHEN a Clarification_Session is created, THE Submission status SHALL transition to "awaiting_clarification".
4. THE Clarification_Session SHALL store the session_id, submission_id, current intent_version, round_count initialized to zero, created_at timestamp, and expires_at timestamp.

### Requirement 2: Clarification Question Generation

**User Story:** As a user, I want to receive specific, structured questions about the ambiguous parts of my prompt, so that I can provide targeted answers without needing to understand the system internals.

#### Acceptance Criteria

1. WHEN a Clarification_Session is created, THE Clarification_Question generator SHALL produce one Clarification_Question per Unresolved_Field in the current CanonicalIntent.
2. THE Clarification_Question SHALL contain the question_id, a human-readable question_text, the intent_path of the Unresolved_Field, the Reason_Code, a list of candidate_options (column names or values from the IntentPackage grounding candidates), and a free_text_enabled flag set to true.
3. THE Clarification_Question SHALL include a "none_of_these" option as the final candidate_option for every question where candidate_options are provided.
4. WHEN the Reason_Code is MULTIPLE_COLUMN_MATCHES, THE candidate_options SHALL be populated from the top-scoring columns in the IntentPackage grounding result, ordered by confidence score descending, and THE question_text SHALL only present the candidate options without describing the column matching ambiguity.
5. WHEN the Reason_Code is AMBIGUOUS_REFERENCE, THE question_text SHALL describe the ambiguity and present the competing interpretations as candidate_options.

### Requirement 3: Clarification Delivery via WebSocket

**User Story:** As a user, I want to receive clarification questions in real-time without refreshing the page, so that I can respond immediately and keep my workflow moving.

#### Acceptance Criteria

1. WHEN Clarification_Questions are generated, THE WebSocket_Manager SHALL broadcast a "clarification_required" event on the "uploads" channel with the session_id, submission_id, intent_version, list of questions, and the Revision_Token.
2. THE "clarification_required" WebSocket event payload SHALL include an expires_at timestamp indicating when the Clarification_Session will expire.
3. WHEN a Clarification_Session expires, THE WebSocket_Manager SHALL broadcast a "clarification_expired" event on the "uploads" channel with the session_id and submission_id.

### Requirement 4: Clarification Response Submission

**User Story:** As a user, I want to submit my answers to clarification questions through a structured API, so that the system can validate my input and proceed with processing.

#### Acceptance Criteria

1. THE Backend SHALL expose a POST endpoint at `/api/uploads/{submission_id}/clarify` that accepts a Clarification_Response payload.
2. THE Clarification_Response payload SHALL require session_id, intent_version, revision_token, and an answers array where each answer contains question_id and either a selected_option or free_text value.
3. WHEN the submitted revision_token does not match the current Revision_Token for the Clarification_Session, THE Backend SHALL reject the request with a 409 Conflict status and a SESSION_EXPIRED Clarification_Outcome.
4. WHEN the submitted intent_version does not match the current intent_version of the Clarification_Session, THE Backend SHALL reject the request with a 409 Conflict status and a stale-version error message.
5. WHEN an answer references a question_id that does not belong to the current Clarification_Session, THE Backend SHALL reject the request with a 400 Bad Request status.

### Requirement 5: Response Validation

**User Story:** As a user, I want the system to validate my clarification answers and give me immediate feedback if my response is invalid, so that I can correct mistakes without consuming a clarification round.

#### Acceptance Criteria

1. WHEN a Clarification_Response contains an answer with a selected_option that is not in the original candidate_options and is not "none_of_these", OR WHEN the selected_option does not exist in the specific question's candidate_options array, THE Backend SHALL return an INVALID_RESPONSE Clarification_Outcome without incrementing the round_count.
2. WHEN a Clarification_Response contains an answer where "none_of_these" is selected and the free_text field is empty, THE Backend SHALL return an INVALID_RESPONSE Clarification_Outcome without incrementing the round_count.
3. WHEN a Clarification_Response passes all validation checks, THE Backend SHALL increment the round_count by one and proceed to intent patching.
4. THE Backend SHALL return the INVALID_RESPONSE Clarification_Outcome with a per-question error_details array identifying which answers failed validation and why.

### Requirement 6: Intent Patching (Selective Reprocessing)

**User Story:** As a user, I want my clarification answers to patch only the ambiguous parts of my existing intent, so that valid parts of my original request are preserved and processing is fast.

#### Acceptance Criteria

1. WHEN a valid Clarification_Response is received, THE Backend SHALL apply an Intent_Patch to only the Unresolved_Fields targeted by the answered questions, preserving all other fields of the CanonicalIntent unchanged.
2. WHEN a user selects a candidate_option, THE Intent_Patch SHALL resolve the corresponding Unresolved_Field by setting the resolved_column to the selected option with confidence 1.0 and resolution_method "user_clarification".
3. WHEN a user selects "none_of_these" and provides free_text, THE Intent_Patch SHALL set the Unresolved_Field raw_reference to the user-provided text and re-run column resolution against the free_text value.
4. AFTER applying the Intent_Patch, THE Backend SHALL increment the intent_revision, compute a new intent_hash, set the parent_intent_id to the previous intent_id, and persist a new CanonicalIntentRevision record.
5. AFTER applying the Intent_Patch, THE Backend SHALL re-run grounding and validation on the patched CanonicalIntent without re-running extraction or normalization.

### Requirement 7: Clarification Round Management

**User Story:** As a user, I want the system to limit the number of clarification rounds so that I am not stuck in an endless loop, while allowing validation retries without penalty.

#### Acceptance Criteria

1. THE Clarification_Session SHALL enforce a maximum of 2 semantic clarification rounds (INVALID_RESPONSE submissions do not count toward this limit).
2. WHEN the re-grounding after an Intent_Patch produces no remaining Unresolved_Fields, THE Clarification_Session SHALL conclude with a RESOLVED Clarification_Outcome and the Submission SHALL proceed to execution plan dispatch.
3. WHEN the re-grounding after an Intent_Patch still contains Unresolved_Fields and the round_count is less than 2, THE Clarification_Session SHALL increment the round_count immediately and generate new Clarification_Questions for the remaining Unresolved_Fields and return a STILL_AMBIGUOUS Clarification_Outcome.
4. WHEN the re-grounding after an Intent_Patch still contains Unresolved_Fields and the round_count equals 2, THE Clarification_Session SHALL conclude with a MAX_ROUNDS_EXCEEDED Clarification_Outcome and the Submission SHALL be quarantined.
5. WHEN the Intent_Patch introduces a conflict (e.g., the same column is referenced in contradictory operations), THE Clarification_Session SHALL return a CONFLICT_INTRODUCED Clarification_Outcome without incrementing the round_count.

### Requirement 8: Intent Versioning

**User Story:** As a developer, I want every clarification-driven intent modification to be tracked as an explicit version, so that I can audit the full resolution history.

#### Acceptance Criteria

1. THE CanonicalIntent SHALL maintain a monotonically increasing intent_revision number, starting at 1 for the initial extraction and incrementing by 1 for each successful Intent_Patch.
2. WHEN an Intent_Patch is applied, THE Backend SHALL persist the previous CanonicalIntent version as a CanonicalIntentRevision record with the original intent_id, intent_revision, and intent_hash.
3. THE Submission model SHALL store the current intent_revision and intent_hash reflecting the latest patched version.
4. WHEN a Clarification_Session concludes with RESOLVED, THE final intent_revision SHALL be the version dispatched for execution.

### Requirement 9: Stale-Response Protection

**User Story:** As a user, I want the system to prevent my outdated answers from corrupting a newer intent version, so that concurrent interactions do not produce inconsistent state.

#### Acceptance Criteria

1. THE Clarification_Session SHALL generate a unique Revision_Token each time new Clarification_Questions are sent to the user.
2. WHEN a Clarification_Response is submitted, THE Backend SHALL verify that the submitted revision_token matches the current active Revision_Token for the session.
3. WHEN the revision_token does not match, THE Backend SHALL reject the submission with a SESSION_EXPIRED Clarification_Outcome and include the current session state in the error response so the frontend can refresh. WHEN the revision_token matches and other validation errors occur, THE Backend SHALL allow the submission to proceed to further processing.
4. WHEN no revision_token is provided in the Clarification_Response, THE Backend SHALL allow the submission to proceed normally without token validation.
5. THE Revision_Token SHALL be a cryptographically random string of at least 32 characters to prevent guessing.

### Requirement 10: Session Expiration

**User Story:** As a system operator, I want clarification sessions to expire after a configurable timeout, so that abandoned sessions do not block job processing indefinitely.

#### Acceptance Criteria

1. THE Clarification_Session SHALL have a configurable expiration timeout (default 30 minutes from session creation).
2. WHEN a Clarification_Session expires without receiving a valid response, THE Submission SHALL be quarantined with a reason of "clarification_session_expired".
3. WHEN the Backend receives a Clarification_Response for an expired session, THE Backend SHALL return a SESSION_EXPIRED Clarification_Outcome with HTTP status 410 Gone, even if the response arrives after expiration but before the periodic cleanup task runs.
4. THE Backend SHALL run a periodic cleanup task that transitions expired Clarification_Sessions to quarantine.

### Requirement 11: Frontend Clarification UI

**User Story:** As a user, I want a clear, accessible UI for answering clarification questions, so that I can resolve ambiguities without confusion.

#### Acceptance Criteria

1. WHEN a "clarification_required" WebSocket event is received for the currently viewed Submission, THE Frontend SHALL render a ClarificationPanel component displaying all questions with their candidate_options as selectable choices.
2. THE ClarificationPanel SHALL display a "None of these" option for each question, which when selected reveals a free-text input field with a placeholder "Describe what you meant". WHEN validation errors are present on a question, THE free-text input SHALL be disabled until the errors are resolved.
3. THE ClarificationPanel SHALL display the round count (e.g., "Round 1 of 2") and the session expiration countdown.
4. WHEN the user submits answers, THE ClarificationPanel SHALL call the POST `/api/uploads/{submission_id}/clarify` endpoint and display the resulting Clarification_Outcome.
5. WHEN the Clarification_Outcome is INVALID_RESPONSE, THE ClarificationPanel SHALL conditionally highlight the invalid answers with per-question error messages based on the current system state and allow resubmission without indicating a round was consumed.
6. WHEN the Clarification_Outcome is RESOLVED, THE ClarificationPanel SHALL display a success message and transition the job view to the running state.
7. THE ClarificationPanel SHALL be keyboard-accessible with proper ARIA labels on all interactive elements.

### Requirement 12: WebSocket Event Integration

**User Story:** As a frontend developer, I want well-defined WebSocket event types for clarification lifecycle changes, so that I can keep the UI synchronized with server state.

#### Acceptance Criteria

1. THE WebSocket_Manager SHALL support the following clarification-related event types: "clarification_required", "clarification_resolved", "clarification_expired", "clarification_round_update".
2. WHEN a Clarification_Session concludes with RESOLVED, THE WebSocket_Manager SHALL broadcast only the "clarification_resolved" event on the "uploads" channel with the submission_id, final intent_version, and Clarification_Outcome, without broadcasting a "clarification_round_update" event.
3. WHEN a new clarification round begins (STILL_AMBIGUOUS outcome), THE WebSocket_Manager SHALL broadcast a "clarification_round_update" event with the updated questions, new Revision_Token, and current round_count.
4. THE WebSocket event payloads SHALL include submission_id and session_id in every clarification-related event to enable frontend routing.

### Requirement 13: Submission Status Integration

**User Story:** As a user, I want to see "awaiting_clarification" as a distinct job status in the submissions list, so that I know which jobs need my attention.

#### Acceptance Criteria

1. THE SubmissionStatus enum SHALL include an "awaiting_clarification" value.
2. WHEN a Submission enters the "awaiting_clarification" status, THE SubmissionsPage SHALL display the submission with a distinct visual indicator and a "Needs your input" label.
3. WHEN the frontend maps workflow statuses, THE mapWorkflowStatus function SHALL map "awaiting_clarification" to a "clarification" display status, and SHALL include validation to ensure this mapping is enforced even if the mapping function encounters unexpected values.
4. THE job detail view SHALL show the ClarificationPanel when the submission status is "awaiting_clarification".

### Requirement 14: Unresolved-Field Tracking

**User Story:** As a developer, I want each clarification question to be traceable to the exact intent path and reason code, so that I can debug resolution failures and improve the pipeline.

#### Acceptance Criteria

1. THE Clarification_Question SHALL store the intent_path as a JSON-path string (e.g., "actions[1].conditions[0].field.raw_reference") identifying the exact location within the CanonicalIntent that is unresolved.
2. THE Clarification_Question SHALL store a Reason_Code from the set: MULTIPLE_COLUMN_MATCHES, AMBIGUOUS_REFERENCE, LOW_CONFIDENCE_SCORE, MISSING_COLUMN, CONFLICTING_EVIDENCE.
3. WHEN a Clarification_Question is generated from an unresolved PredicateGroundingResult clause, THE intent_path SHALL reference the corresponding filter condition path and the Reason_Code SHALL be derived from the grounding failure reason.
4. THE Clarification_Session SHALL persist the full question-answer history including intent_paths, reason_codes, user selections, and resulting Clarification_Outcomes for each round.
