# Requirements Document

## Introduction

This feature introduces an interactive clarification loop to the FinFlow Agent Service. When the semantic pipeline determines that a user's prompt is ambiguous (unresolved column references, missing requirements, or conflicting instructions), the system will present structured clarification questions to the user via the frontend instead of quarantining the job. The user can respond with answers, the backend patches the intent, and re-runs the pipeline. This loop repeats for a maximum of 3 rounds before falling back to quarantine.

## Glossary

- **Semantic_Pipeline**: The backend service that extracts, normalizes, grounds, and compiles user intent from raw prompts into canonical actions.
- **Clarification_Session**: A bounded interaction cycle between the backend and frontend where the system asks structured questions and the user provides answers to resolve ambiguities.
- **Clarification_Question**: A structured data object representing one question the system needs answered, including question type, options (if applicable), and context.
- **Clarification_Response**: The user's answer to one or more Clarification_Questions within a single round.
- **Clarification_Round**: One complete ask-and-answer cycle within a Clarification_Session.
- **Question_Type**: The category of clarification needed — one of: column_disambiguation, missing_information, conflicting_requirements, or general_ambiguity.
- **WebSocket_Manager**: The backend service (ws_manager) that broadcasts real-time events to connected frontend clients over WebSocket channels.
- **Submission**: The database model representing a user-uploaded file and associated processing job.
- **Intent_Revision**: A versioned snapshot of the canonical intent for a Submission, tracked for auditability.
- **Frontend_Client**: The React 19 web application that renders clarification UI and collects user responses.

## Requirements

### Requirement 1: Detect and Initiate Clarification

**User Story:** As a user, I want the system to ask me clarifying questions when my prompt is ambiguous, so that my job can proceed without me having to rephrase from scratch.

#### Acceptance Criteria

1. WHEN the Semantic_Pipeline returns a resolution_status of "needs_clarification", THE Backend SHALL create a new Clarification_Session linked to the Submission.
2. WHEN a Clarification_Session is created, THE Backend SHALL set the Submission status to "awaiting_clarification" instead of "quarantined".
3. WHEN a Clarification_Session is created, THE Backend SHALL generate structured Clarification_Questions from the SemanticExtractionResult evidence, unresolved_references, and missing_requirements data.
4. WHEN a Clarification_Session is created, THE WebSocket_Manager SHALL broadcast a "clarification_needed" event on the "uploads" channel containing the session ID and list of questions.

### Requirement 2: Structured Clarification Question Generation

**User Story:** As a user, I want clarification questions to be specific and actionable, so that I can quickly provide the information the system needs.

#### Acceptance Criteria

1. WHEN unresolved column references exist in the grounded intent, THE Backend SHALL generate a Clarification_Question of Question_Type "column_disambiguation" containing the user_term, the list of candidate columns, and the context in which the term was used.
2. WHEN missing requirements are detected by coverage verification, THE Backend SHALL generate a Clarification_Question of Question_Type "missing_information" containing the requirement description and source_text from the user prompt.
3. WHEN conflicting requirements are detected, THE Backend SHALL generate a Clarification_Question of Question_Type "conflicting_requirements" containing descriptions of both conflicting tasks and asking the user to choose or clarify priority.
4. WHEN the SemanticIntent contains ambiguities, THE Backend SHALL generate a Clarification_Question of Question_Type "general_ambiguity" containing the ambiguity description and possible_interpretations as selectable options.
5. THE Backend SHALL include a maximum of 5 Clarification_Questions per Clarification_Round to avoid overwhelming the user.

### Requirement 3: Frontend Clarification UI Rendering

**User Story:** As a user, I want to see clarification questions in an intuitive interface with appropriate input controls, so that I can respond quickly and accurately.

#### Acceptance Criteria

1. WHEN the Frontend_Client receives a "clarification_needed" WebSocket event, THE Frontend_Client SHALL display a clarification panel within the job detail view.
2. WHEN a Clarification_Question has Question_Type "column_disambiguation", THE Frontend_Client SHALL render a radio button group showing each candidate column with the original user_term highlighted.
3. WHEN a Clarification_Question has Question_Type "missing_information", THE Frontend_Client SHALL render a text input field with the requirement description as a contextual label.
4. WHEN a Clarification_Question has Question_Type "conflicting_requirements", THE Frontend_Client SHALL render a selectable card group presenting each conflicting interpretation as a distinct choice.
5. WHEN a Clarification_Question has Question_Type "general_ambiguity", THE Frontend_Client SHALL render a selectable list of possible_interpretations with an optional free-text "Other" field.
6. THE Frontend_Client SHALL display the current round number and maximum rounds remaining in the clarification panel header.
7. THE Frontend_Client SHALL provide a "Submit Answers" button that is enabled only when all required questions have responses.

### Requirement 4: Submit Clarification Response

**User Story:** As a user, I want to submit my clarification answers and have the system immediately re-process my job, so that I experience minimal delay.

#### Acceptance Criteria

1. WHEN the user submits a Clarification_Response, THE Frontend_Client SHALL send the response to the Backend via a REST API endpoint `POST /api/uploads/{upload_id}/clarify`.
2. WHEN the Backend receives a Clarification_Response, THE Backend SHALL validate that all required questions have been answered.
3. IF a Clarification_Response is missing required answers, THEN THE Backend SHALL return a 422 response with details of which questions remain unanswered.
4. WHEN a valid Clarification_Response is received, THE Backend SHALL store the response in the Clarification_Session record for audit purposes.
5. WHEN a valid Clarification_Response is received, THE Backend SHALL patch the SemanticIntent using the user's answers (resolving column references, adding missing task parameters, or removing conflicting tasks).
6. WHEN the intent is patched, THE Backend SHALL re-run the Semantic_Pipeline from the grounding step with the updated intent.

### Requirement 5: Iterative Resolution with Bounded Rounds

**User Story:** As a user, I want the system to re-evaluate after each set of answers and either proceed or ask follow-up questions, so that complex ambiguities are resolved incrementally.

#### Acceptance Criteria

1. WHEN the re-run of the Semantic_Pipeline succeeds (resolution_status is "resolved" or "repaired"), THE Backend SHALL persist a new Intent_Revision, set Submission status to "queued", and dispatch the job to the agent service.
2. WHEN the re-run of the Semantic_Pipeline still returns "needs_clarification" and the current round is less than 3, THE Backend SHALL generate new Clarification_Questions and broadcast a new "clarification_needed" event for the next round.
3. WHEN the re-run of the Semantic_Pipeline still returns "needs_clarification" and the current round equals 3, THE Backend SHALL set the Submission status to "quarantined" and broadcast a "clarification_exhausted" event with a message explaining that manual review is required.
4. THE Backend SHALL track the current round number on the Clarification_Session and increment it after each Clarification_Response is processed.
5. WHILE a Clarification_Session is active (rounds remaining and status is "awaiting_clarification"), THE Submission SHALL remain in "awaiting_clarification" status and not be picked up by the worker queue.

### Requirement 6: Real-Time Status Updates During Clarification

**User Story:** As a user, I want to see real-time feedback when my clarification is being processed, so that I know the system is working on my job.

#### Acceptance Criteria

1. WHEN a Clarification_Response is received and processing begins, THE WebSocket_Manager SHALL broadcast a "clarification_processing" event on the "uploads" channel with the submission ID.
2. WHEN clarification processing completes successfully, THE WebSocket_Manager SHALL broadcast a "clarification_resolved" event containing the new Submission status and job progress.
3. WHEN clarification processing results in another round, THE WebSocket_Manager SHALL broadcast a "clarification_needed" event with the new questions and updated round number.
4. THE Frontend_Client SHALL display a processing indicator between submission of answers and receipt of the next WebSocket event.

### Requirement 7: Clarification Session Persistence and Audit Trail

**User Story:** As a system administrator, I want a complete audit trail of clarification interactions, so that I can review how ambiguities were resolved and improve the system.

#### Acceptance Criteria

1. THE Backend SHALL persist each Clarification_Session with: session_id, submission_id, created_at, status (active, resolved, exhausted), max_rounds, and current_round.
2. THE Backend SHALL persist each Clarification_Round within a session with: round_number, questions (as structured JSON), user_response (as structured JSON), submitted_at, and resolution_outcome (resolved, next_round, exhausted).
3. WHEN a Clarification_Session completes (either resolved or exhausted), THE Backend SHALL record the final outcome and timestamp on the session record.
4. THE Backend SHALL link the Clarification_Session to the Intent_Revision that was produced after successful resolution.

### Requirement 8: Clarification Timeout Handling

**User Story:** As a system operator, I want clarification sessions that remain unanswered to expire gracefully, so that stale jobs do not accumulate indefinitely.

#### Acceptance Criteria

1. WHILE a Clarification_Session has been in "active" status for more than 24 hours without a user response, THE Backend SHALL set the session status to "expired" and the Submission status to "quarantined".
2. WHEN a Clarification_Session expires, THE WebSocket_Manager SHALL broadcast a "clarification_expired" event so the Frontend_Client can display an appropriate message.
3. IF a user attempts to submit a Clarification_Response to an expired session, THEN THE Backend SHALL return a 410 Gone response indicating the session has expired and the user should resubmit the job.

### Requirement 9: Integration with Existing Workflow States

**User Story:** As a user, I want the clarification state to integrate seamlessly with the existing job tracking UI, so that I can find and respond to jobs needing clarification from my submissions list.

#### Acceptance Criteria

1. THE Frontend_Client SHALL map the "awaiting_clarification" Submission status to a distinct visual state in the submissions list (separate from "quarantined" and "running").
2. WHEN a Submission is in "awaiting_clarification" status, THE Frontend_Client SHALL display a "Respond" action button on the submissions list that navigates to the job detail clarification panel.
3. THE Backend SHALL include "awaiting_clarification" in the status filter options for the submissions list API endpoint.
4. WHEN the Submission status transitions from "awaiting_clarification" to "queued" (after successful resolution), THE Frontend_Client SHALL update the submissions list in real-time via WebSocket cache invalidation.
