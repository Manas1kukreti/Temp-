# Requirements Document

## Introduction

This document specifies the requirements for refactoring the FinFlow system's natural-language interpretation and grounding architecture into a stable, auditable, multi-stage pipeline with clearly separated responsibilities. The current system suffers from premature semantic commitment, destroyed boolean scope, overlapping column resolution authorities, unreliable LLM-based coverage checking, unbounded repair, and late grounding. The refactored architecture introduces SemanticIntentDraft as a pre-canonical model, enforces single-owner decision authority per semantic element, moves grounding before compilation, removes LLM from the authoritative coverage path, and guarantees that no unresolved reference reaches execution.

## Glossary

- **Semantic_Pipeline**: The end-to-end processing chain from raw user prompt to finalized CanonicalIntent, encompassing extraction, validation, repair, grounding, and compilation stages.
- **SemanticIntentDraft**: A pre-canonical intermediate representation that preserves ambiguities, unresolved references, and source provenance. Each draft is an immutable revision; modifications produce a new revision. May hold any resolution status including pending, needs_clarification, unsupported, invalid, or resolved.
- **CanonicalIntent**: The finalized, fully-resolved, executable intent representation. Created only after all grounding, clarification, and validation completes. Always has resolution_status "resolved". Contains no unresolved active execution references. Historical resolution metadata may be retained in immutable audit fields.
- **IntentEnvelope**: The container that tracks pipeline state: holds either a SemanticIntentDraft (during resolution) or a CanonicalIntent (after finalization), plus pipeline status and metadata.
- **Intent_Resolution_Coordinator**: The explicit component that finalizes operation classification and boolean scope. Receives operation candidates from extraction, applies deterministic validation results, applies user clarification patches, selects a unique operation when supported, and preserves unresolved ambiguity otherwise.
- **Semantic_Extractor**: The LLM-based stage that converts a raw user prompt into a SemanticIntentDraft, including typed provenance references for every extracted element.
- **SemanticColumnReference**: A typed column reference within SemanticIntentDraft that classifies the reference kind (explicit_name, semantic_concept, generic_reference, value_implied, column_group) and retains provenance references.
- **Column_Grounder**: The deterministic-first component responsible for resolving standalone column references (projections, sorts, renames) to physical columns using the Candidate_Generation_Layer and optional LLM fallback with post-LLM deterministic verification.
- **Predicate_Grounder**: The deterministic-first component responsible for resolving complete filter predicates (field + operator + value) to physical columns, owning the final filter-column decision; uses the Candidate_Generation_Layer and optional LLM fallback with post-LLM deterministic verification.
- **Coverage_Validator**: The deterministic component that verifies structural correctness, provenance traceability, and action-to-reference completeness of the extracted intent — without re-interpreting prompt semantics independently.
- **Semantic_Repair**: The bounded LLM-based component that applies typed patch sets against declared draft paths (add, replace, remove operations only); maximum one attempt per pipeline invocation.
- **Schema_Service**: The component that infers column roles and semantic types from dataset profiling. Uses a layered cache: structural role cache keyed by Structural_Schema_Fingerprint and role-model version; value-evidence cache keyed by Structural_Schema_Fingerprint and Profile_Fingerprint.
- **Compiler**: The deterministic component that transforms a finalized CanonicalIntent into an ExecutionPlan; only accepts CanonicalIntent (which is always resolved by type-level guarantee).
- **Executor**: The deterministic engine that walks an ExecutionPlan; performs no LLM calls, no grounding, and no semantic interpretation.
- **Clarification_Service**: The interactive component that presents ambiguity options to users and patches the existing SemanticIntentDraft with user responses, producing a new immutable draft revision.
- **Candidate_Generation_Layer**: The shared deterministic subsystem used by both Column_Grounder and Predicate_Grounder to produce scored candidate columns from semantic profiles.
- **Structural_Schema_Fingerprint**: A deterministic hash of the dataset's structural schema: normalized column names, column order, inferred physical dtypes, nullable information, and profiler version. Used for schema compatibility identification and structural role cache key.
- **Profile_Fingerprint**: A deterministic hash of semantic statistics, cardinality buckets, representative-value hashes, and profiling configuration. Used when value evidence affects the semantic proposal. Avoids hashing raw values that may contain sensitive information.
- **DataSnapshotRef**: The immutable reference identifying a profiled file version: file_id, content_hash, byte_size, storage_version, profile_id, structural_schema_fingerprint, and profile_fingerprint.
- **Decision_Owner**: The single component that has final authority over a specific semantic decision; no other component may override a decision once its owner has finalized it.
- **Preflight_Grounding**: The architectural requirement that all column/predicate grounding completes before compilation begins.
- **Preflight_Data_Loader**: The read-only component that loads and profiles the source file before compilation, producing a reusable DataFrameProfile and DataSnapshotRef without performing any transformations.
- **Shadow_Mode**: An operational mode where a component runs in parallel but its output has no authority over the pipeline result; used for LLM coverage comparison.
- **Ambiguity_Margin**: The configurable threshold (default 0.1) below which two candidate scores are considered too close to resolve without clarification.
- **Confidence_Threshold**: The minimum score (default 0.75) below which a grounding result is not trusted and requires LLM fallback or clarification.
- **PromptSpanProvenance**: A provenance annotation recording start offset (original-prompt Unicode code-point offset), end offset, and extracted text for a semantic element sourced from the raw prompt.
- **ClarificationProvenance**: A provenance annotation recording question_id, response_id, and selected_value for a semantic element sourced from user clarification.
- **SchemaEvidenceProvenance**: A provenance annotation recording schema_fingerprint, column, and evidence list for a semantic element inferred from schema evidence.
- **ProvenanceRef**: The union type of PromptSpanProvenance, ClarificationProvenance, and SchemaEvidenceProvenance; each semantic element references one or more ProvenanceRef entries. A single source span may support multiple related semantic elements (many-to-many relationship).
- **SemanticPatch**: A typed operation (add, replace, remove) applied against a declared draft path, including reason and optional ProvenanceRef.
- **Draft_Revision**: An immutable, monotonically-incrementing version number assigned to each SemanticIntentDraft instance. Modifications produce a new revision; previous revisions are never mutated.
- **Resolution_Status**: The validity state of a draft or intent: pending, needs_clarification, interpretation_failed, unsupported, invalid, or resolved.
- **Resolution_Origin**: The workflow path that produced the current resolution: direct, semantic_repair, automatic_grounding, or user_clarification.
- **Pipeline_Status**: The operational status of the pipeline: processing, needs_clarification, interpretation_failed, unsupported, invalid, or resolved. Distinguishes provider failure (interpretation_failed) from genuine user ambiguity (needs_clarification).

## Requirements

### Requirement 1: SemanticIntentDraft as Pre-Canonical Model

**User Story:** As a pipeline architect, I want the extraction stage to produce a draft representation that preserves ambiguities, unresolved references, and typed provenance, so that downstream stages can make informed grounding decisions without premature commitment.

#### Acceptance Criteria

1. WHEN the Semantic_Extractor processes a raw prompt, THE Semantic_Extractor SHALL produce a SemanticIntentDraft that allows unresolved column references, ambiguity markers, and typed ProvenanceRef entries for every extracted element.
2. THE SemanticIntentDraft SHALL include a SemanticColumnReference for each column mention, with a reference_kind classified as one of: explicit_name, semantic_concept, generic_reference, value_implied, or column_group.
3. WHEN a prompt contains generic words (field, column, attribute, entry, data), THE Semantic_Extractor SHALL classify them as generic_reference rather than treating them as literal column names.
4. WHEN multiple interpretations of an operation are plausible, THE Semantic_Extractor SHALL preserve all plausible interpretations in the SemanticIntentDraft ambiguity markers rather than committing to a single classification.
5. WHEN a prompt contains value-list patterns (e.g., "paypal or cash"), THE Semantic_Extractor SHALL preserve the boolean scope as a value set within one predicate rather than splitting into separate clauses.
6. THE SemanticIntentDraft SHALL include at least one ProvenanceRef for each extracted action, reference, operator, value, logical group, and exclusion. A single source span MAY support multiple related semantic elements.

### Requirement 2: Single Decision-Ownership Authority

**User Story:** As a system maintainer, I want each semantic decision to have exactly one final owner defined in a formal ownership matrix, so that competing authorities cannot produce conflicting resolutions.

#### Acceptance Criteria

1. THE Semantic_Pipeline SHALL assign exactly one Decision_Owner to each semantic element (column reference, operator, value, action classification) as defined in the decision-ownership matrix.
2. THE decision-ownership matrix SHALL define: operation classification owned by Intent_Resolution_Coordinator; boolean scope owned by Intent_Resolution_Coordinator; standalone column resolution owned by Column_Grounder; filter column resolution owned by Predicate_Grounder; canonical operator mapping owned by Predicate_Grounder; predicate value normalization owned by Predicate_Grounder; dataset semantic role owned by Schema_Service; ambiguous user choice owned by User via Clarification_Service; execution step selection owned by Compiler.
3. WHEN both Column_Grounder and Predicate_Grounder could resolve the same reference, THE Semantic_Pipeline SHALL route the reference to exactly one grounder based on whether the reference is a standalone column reference or part of a complete filter predicate.
4. THE Column_Grounder SHALL own final resolution authority for standalone column references (projections, sorts, drops, renames) and SHALL NOT resolve filter predicate columns.
5. THE Predicate_Grounder SHALL own final resolution authority for complete filter predicates (field + operator + value) and SHALL NOT delegate filter-column decisions to the Column_Grounder.
6. IF a Decision_Owner has finalized a resolution, THEN THE Semantic_Pipeline SHALL reject any subsequent attempt by another component to override that resolution.

### Requirement 3: Deterministic Coverage Validation

**User Story:** As a reliability engineer, I want coverage verification to check structural correctness and provenance traceability without re-interpreting prompt semantics, so that the validator does not become a competing semantic authority.

#### Acceptance Criteria

1. THE Coverage_Validator SHALL use deterministic structural checks as the authoritative coverage boundary, verifying: JSON and schema validity; action-to-reference completeness; negation and boolean-group preservation from extractor-provided spans; duplicate and contradictory action detection; unresolved-reference declarations; valid contract tokens; and every extracted element linked to at least one ProvenanceRef.
2. THE Coverage_Validator SHALL verify that every material source span identified by extraction is linked to at least one semantic element, ambiguity record, or explicitly ignored-span record; shared spans MAY support multiple related semantic elements.
3. THE Coverage_Validator SHALL NOT independently re-interpret the raw prompt using keyword analysis or pattern matching to determine semantic meaning.
4. WHEN LLM coverage checking is enabled, THE Coverage_Validator SHALL run LLM coverage in Shadow_Mode only, behind a feature flag, with no authority over pipeline progression.
5. THE Coverage_Validator SHALL emit a structured comparison metric between deterministic and LLM coverage results when Shadow_Mode is active.
6. IF the Coverage_Validator detects a structural gap, THEN THE Semantic_Pipeline SHALL route to Semantic_Repair with the specific declared validation failure.

### Requirement 4: Bounded Semantic Repair

**User Story:** As a pipeline architect, I want semantic repair to apply typed patch sets against declared draft paths, so that repair cannot become unrestricted re-extraction.

#### Acceptance Criteria

1. THE Semantic_Repair SHALL accept only declared patch paths corresponding to specific Coverage_Validator structural failures.
2. THE Semantic_Repair SHALL execute a maximum of one repair attempt per pipeline invocation.
3. THE Semantic_Repair SHALL return a typed patch set (SemanticPatch) containing only add, replace, or remove operations against declared semantic draft paths, not a complete re-extraction.
4. THE Semantic_Pipeline SHALL apply SemanticPatch operations deterministically against the current SemanticIntentDraft, producing a new Draft_Revision.
5. IF the Coverage_Validator still reports structural gaps after one repair attempt, THEN THE Semantic_Pipeline SHALL fail closed with a needs_clarification status rather than retrying.
6. THE Semantic_Repair SHALL log the declared patch path, input failure, patch operation type, and resulting modification for observability.

### Requirement 5: Schema Service Layered Caching

**User Story:** As a performance engineer, I want schema proposals to be cached in layers (structural roles vs. value evidence) with stable invalidation behavior, so that repeated computations are eliminated without spurious cache misses when only data distribution changes.

#### Acceptance Criteria

1. THE Schema_Service SHALL maintain a structural role cache keyed by Structural_Schema_Fingerprint and role-model version, caching role inferences that depend only on column structure.
2. THE Schema_Service SHALL maintain a value-evidence cache keyed by Structural_Schema_Fingerprint, Profile_Fingerprint, and profiler version, caching semantic proposals that depend on observed values.
3. WHEN a schema inference request arrives with a matching structural fingerprint and role-model version, THE Schema_Service SHALL return the cached structural role result without re-computation.
4. WHEN the dataset structural schema changes (columns added, removed, or types changed), THE Schema_Service SHALL compute a new Structural_Schema_Fingerprint; previous cache entries SHALL be retained under their original keys (not globally invalidated) and expired according to policy.
5. THE Schema_Service SHALL execute schema inference during the dataset-profiling stage, after the Preflight_Data_Loader completes and before grounding begins.
6. THE Schema_Service SHALL NOT hash raw values that may contain sensitive information when computing the Profile_Fingerprint.

### Requirement 6: Preflight Grounding (Grounding Before Compilation)

**User Story:** As a pipeline architect, I want all grounding to complete before compilation, so that no unresolved active execution reference can reach execution.

#### Acceptance Criteria

1. THE Semantic_Pipeline SHALL complete all column and predicate grounding before invoking the Compiler.
2. THE Compiler SHALL only accept CanonicalIntent objects, which by type-level guarantee contain no unresolved active execution references.
3. THE Executor SHALL reject any ExecutionPlan that references columns not present in the validated intent package, returning a fail-closed error.
4. THE Semantic_Pipeline SHALL NOT perform any grounding during or after the compilation stage.
5. IF grounding produces unresolved references below the Confidence_Threshold with no viable LLM fallback, THEN THE Semantic_Pipeline SHALL route to Clarification_Service rather than allowing low-confidence execution.
6. THE CanonicalIntent MAY retain historical resolution metadata (resolution_history, provenance) in immutable audit fields while containing no unresolved active execution references.

### Requirement 7: Shared Candidate Generation Layer

**User Story:** As a developer, I want both grounders to share a common candidate-generation subsystem, so that scoring logic is consistent and maintainable.

#### Acceptance Criteria

1. THE Candidate_Generation_Layer SHALL provide a shared interface for both Column_Grounder and Predicate_Grounder to generate scored candidate columns from semantic profiles.
2. THE Candidate_Generation_Layer SHALL produce candidates using deterministic scoring based on token overlap, value-concept matching, semantic-type alignment, and column-name similarity.
3. WHEN both grounders request candidates for the same column, THE Candidate_Generation_Layer SHALL return identical scores for identical inputs.
4. THE Candidate_Generation_Layer SHALL expose candidate scores, positive evidence, and negative evidence for each candidate to support observability.

### Requirement 8: LLM Call Disposition, Constraints, and Post-LLM Verification

**User Story:** As a system architect, I want each LLM call site to have a defined role, constraint, failure behavior, post-LLM verification, and explicit tie-breaking policy, so that LLM usage is auditable and bounded.

#### Acceptance Criteria

1. THE Semantic_Extractor LLM call SHALL output a SemanticIntentDraft (not a CanonicalIntent) and SHALL preserve ambiguity rather than guessing; it SHALL include typed ProvenanceRef entries for every extracted element.
2. THE Semantic_Repair LLM call SHALL operate as a bounded patch (maximum one attempt) returning only typed SemanticPatch operations against declared paths.
3. THE Schema_Service LLM call SHALL be cached by Structural_Schema_Fingerprint and Profile_Fingerprint and SHALL execute during the dataset-profiling stage.
4. THE Column_Grounder LLM fallback SHALL resolve only standalone column references and SHALL constrain selection to existing physical columns.
5. THE Predicate_Grounder LLM fallback SHALL resolve only complete filter predicates and SHALL constrain selection to existing physical columns.
6. THE Executor SHALL make zero LLM calls during plan execution.
7. WHEN an LLM fallback returns a selection, THE Semantic_Pipeline SHALL apply post-LLM deterministic verification: validate selected candidate exists, validate operator compatibility, validate value shape, validate dtype compatibility, validate candidate was in the permitted set, and apply the tie-breaking policy before accepting the result.
8. IF post-LLM deterministic verification fails or deterministic evidence strongly conflicts with the LLM selection, THEN THE Semantic_Pipeline SHALL route to Clarification_Service rather than accepting the LLM result.
9. THE tie-breaking policy SHALL be: LLM agrees with strong deterministic leader (margin above Ambiguity_Margin) → may resolve; LLM is the only evidence breaking a close tie (within Ambiguity_Margin) → clarification required; LLM conflicts with deterministic leader → clarification required; destructive operation with close candidates → clarification required regardless of LLM selection.

### Requirement 9: Clarification as Draft Patching with Stage-Resume Policy

**User Story:** As a user, I want the system to ask me for clarification when ambiguous rather than guessing, and I want my response to patch the existing draft and resume from the earliest affected stage.

#### Acceptance Criteria

1. WHEN an ambiguity triggers clarification, THE Clarification_Service SHALL present candidate options derived from the authoritative ambiguity source: semantic extraction candidates for operation and scope ambiguities, Candidate_Generation_Layer results for column ambiguities, and predicate-grounding evidence for operator or value ambiguities.
2. WHEN a user responds to a clarification prompt, THE Clarification_Service SHALL patch the existing SemanticIntentDraft with the user's selection, producing a new immutable Draft_Revision rather than re-extracting from the original prompt.
3. THE Clarification_Service SHALL apply a stage-resume policy based on the patched path: column clarification resumes from affected grounding; operation clarification resumes from semantic validation and affected grounding; value clarification resumes from predicate grounding; prompt replacement restarts from extraction.
4. THE Clarification_Service SHALL trigger clarification when: filter vs projection is ambiguous, multiple operation interpretations are plausible, multiple columns are within the Ambiguity_Margin, a generic reference has no unique match, values appear in multiple columns, schema evidence conflicts with value evidence, operator scope is unclear, confidence is below Confidence_Threshold, LLM selection conflicts with deterministic evidence, or a destructive action rests on weak evidence.

### Requirement 10: Resolution Policies

**User Story:** As a system architect, I want deterministic policies for each class of uncertainty, so that low-confidence grounding never silently executes.

#### Acceptance Criteria

1. WHEN formatting uncertainty occurs, THE Semantic_Pipeline SHALL emit a warning and proceed with the best-available formatting.
2. WHEN semantic column uncertainty occurs (multiple plausible columns within Ambiguity_Margin), THE Semantic_Pipeline SHALL route to Clarification_Service.
3. WHEN operation ambiguity occurs (filter vs projection unclear), THE Semantic_Pipeline SHALL route to Clarification_Service.
4. WHEN no physical column matches a reference, THE Semantic_Pipeline SHALL classify the reference as unsupported or route to Clarification_Service.
5. WHEN a contract violation is detected (e.g., unresolved reference reaching compilation), THE Semantic_Pipeline SHALL fail closed or quarantine the submission.
6. THE Semantic_Pipeline SHALL NOT allow low-confidence grounding to execute with only a warning; low-confidence resolution SHALL either clarify or fail.

### Requirement 11: Execution Boundary Contracts

**User Story:** As a reliability engineer, I want strict contracts at the compilation and execution boundaries, so that partially-resolved intents cannot cause runtime failures.

#### Acceptance Criteria

1. THE Compiler SHALL validate that every column reference in the CanonicalIntent is resolved to a physical column before emitting an ExecutionPlan.
2. THE Compiler SHALL accept only CanonicalIntent objects; the type-level guarantee ensures resolution_status is always "resolved".
3. THE Executor SHALL validate that every column referenced in plan steps exists in the intent package before executing.
4. IF the Executor receives a plan step referencing a column not present in the intent package, THEN THE Executor SHALL return a fail-closed error without executing the step.
5. THE Executor SHALL perform zero LLM calls, zero grounding operations, and zero semantic interpretations during execution.

### Requirement 12: Observability and Structured Tracing

**User Story:** As an operations engineer, I want structured metrics and tracing for every pipeline stage, so that I can monitor health, debug failures, and measure improvement.

#### Acceptance Criteria

1. THE Semantic_Pipeline SHALL emit metrics for: extraction attempts, extraction successes, extraction failures (including interpretation_failed), grounding attempts, grounding successes, grounding LLM fallback invocations, post-LLM verification passes, post-LLM verification failures, clarification sessions initiated, clarification sessions resolved, repair attempts, repair successes, coverage check passes, and coverage check failures.
2. THE Semantic_Pipeline SHALL include structured tracing fields: submission_id, draft_id, draft_revision, intent_id, schema_fingerprint, profile_fingerprint, data_snapshot_ref, model_version, pipeline_stage, decision_owner, and duration_ms on every log entry.
3. WHEN LLM Shadow_Mode is active, THE Coverage_Validator SHALL emit a comparison metric recording deterministic_result, llm_result, and agreement_status.
4. THE Semantic_Pipeline SHALL record the Decision_Owner for each finalized semantic element in the structured trace.

### Requirement 13: Migration and Feature-Flag Support

**User Story:** As a deployment engineer, I want the refactoring to be deployed incrementally behind feature flags with validated compatibility constraints, so that rollback is possible at each stage without conflicting authorities.

#### Acceptance Criteria

1. THE Semantic_Pipeline SHALL support feature flags to enable or disable: SemanticIntentDraft output, preflight grounding, LLM coverage Shadow_Mode, bounded repair paths, schema caching, and clarification-as-patching.
2. WHILE a feature flag is disabled, THE Semantic_Pipeline SHALL use the legacy behavior for the corresponding stage, maintaining backward compatibility.
3. THE Semantic_Pipeline SHALL validate feature-flag combinations at application startup: ENABLE_PREFLIGHT_GROUNDING requires DISABLE_EXECUTION_TIME_GROUNDING; ENABLE_SEMANTIC_DRAFT_PIPELINE is required for clarification-as-draft-patching; conflicting combinations SHALL cause startup failure with a descriptive error.
4. THE Semantic_Pipeline SHALL support versioned upcasters that convert legacy CanonicalIntent objects to the new schema format during the migration period.
5. WHEN a feature flag is toggled, THE Semantic_Pipeline SHALL log the transition with the previous state, new state, and effective timestamp.
6. THE Semantic_Pipeline SHALL NOT permit a mixed mode where both legacy execution-time grounding and preflight grounding are active for the same reference type.

### Requirement 14: SemanticIntentDraft Schema and Serialization

**User Story:** As a developer, I want SemanticIntentDraft to be reliably serializable, versioned, and extensible, so that drafts can be persisted, transmitted, and reconstructed without data loss or silent coercion.

#### Acceptance Criteria

1. THE SemanticIntentDraft SHALL be serializable to JSON via Pydantic model_dump(mode="json").
2. THE SemanticIntentDraft SHALL be deserializable from JSON via model_validate(json_data).
3. FOR ALL valid SemanticIntentDraft objects, serializing then deserializing SHALL produce an equivalent object (round-trip property).
4. THE SemanticIntentDraft SHALL include a schema_version field; the pipeline SHALL reject draft objects with unsupported future schema versions.
5. THE SemanticIntentDraft SHALL use discriminated action unions for intent actions to ensure unambiguous deserialization.
6. THE SemanticIntentDraft SHALL avoid silent Pydantic coercion that could alter semantic meaning; strict mode or explicit validators SHALL be used for fields where coercion would change intent.
7. THE SemanticIntentDraft pretty-printer SHALL format draft objects into human-readable structured text for debugging and logging.

### Requirement 15: Source Provenance and Traceability

**User Story:** As a debugging engineer, I want every extracted semantic element to retain typed, many-to-many provenance to its source, so that coverage validation and conflict diagnosis can be grounded in evidence.

#### Acceptance Criteria

1. THE Semantic_Extractor SHALL produce at least one PromptSpanProvenance for every extracted action, reference, operator, value, logical group, and exclusion, recording start offset (original-prompt Unicode code-point offset), end offset, and source text.
2. THE Coverage_Validator SHALL verify that every extracted semantic element has at least one valid ProvenanceRef.
3. THE Coverage_Validator SHALL verify that all material source spans are linked to at least one semantic element, ambiguity record, or explicitly ignored-span record; shared spans MAY support multiple related semantic elements.
4. WHEN a Semantic_Repair patch adds or modifies elements, THE patch SHALL include a ProvenanceRef (PromptSpanProvenance or SchemaEvidenceProvenance) for the new or modified element.
5. THE Clarification_Service SHALL record user selections as ClarificationProvenance (with question_id, response_id, and selected_value) rather than as synthetic prompt spans.

### Requirement 16: Preflight Data Access and Profile Lifecycle

**User Story:** As a pipeline architect, I want source data to be profiled against an immutable file snapshot with the profile persisted for reuse, so that grounding has value evidence without causing inconsistent reads between profiling and execution.

#### Acceptance Criteria

1. THE Preflight_Data_Loader SHALL load the source file and produce a DataFrameProfile and a DataSnapshotRef before the Schema_Service and grounding stages execute.
2. THE Preflight_Data_Loader SHALL operate in read-only mode; it SHALL NOT perform any cleaning, transformation, or mutation of the source data.
3. THE Preflight_Data_Loader SHALL compute a content_hash (file-content fingerprint) as part of the DataSnapshotRef so that the pipeline can verify the file has not changed between profiling and execution.
4. IF the file content_hash at execution time differs from the DataSnapshotRef content_hash, THEN THE Executor SHALL fail closed with a data-consistency error.
5. THE Semantic_Pipeline SHALL persist and reuse the DataFrameProfile produced by the Preflight_Data_Loader; the Executor MAY reload the immutable source file after verifying its content_hash but SHALL NOT re-profile it.
6. THE Preflight_Data_Loader SHALL enforce configurable size limits; files exceeding the limit SHALL be rejected before profiling with a descriptive error.

### Requirement 17: Revision and Concurrency Safety

**User Story:** As a reliability engineer, I want SemanticIntentDraft modifications to be protected against concurrent writes and stale responses, so that clarification does not corrupt shared state.

#### Acceptance Criteria

1. THE SemanticIntentDraft SHALL include a draft_revision field that increments monotonically with each modification; previous revisions SHALL be immutable.
2. WHEN the Clarification_Service applies a user response, THE Clarification_Service SHALL include the expected_revision in the patch request; if the current draft revision differs, THE Clarification_Service SHALL reject the response as stale.
3. THE Clarification_Service SHALL include a question_id and idempotency_key with each clarification response to prevent duplicate applications.
4. IF a clarification response targets a superseded Draft_Revision, THEN THE Clarification_Service SHALL return a stale-response error and present the user with the updated clarification state.

### Requirement 18: LLM Failure and Degradation Policy

**User Story:** As a reliability engineer, I want explicit failure handling for each LLM call site that distinguishes provider failure from genuine user ambiguity, so that the system degrades gracefully without asking users questions that arise from system errors.

#### Acceptance Criteria

1. WHEN the Semantic_Extractor LLM is unavailable (timeout, rate limit, invalid JSON, schema validation failure, empty output, or provider outage), THE Semantic_Pipeline SHALL retry according to a bounded provider policy and then set Pipeline_Status to interpretation_failed; it SHALL NOT fall back to deterministic keyword guessing or set needs_clarification.
2. WHEN the Schema_Service LLM is unavailable, THE Schema_Service SHALL return a cached schema if a compatible Structural_Schema_Fingerprint exists; otherwise THE Schema_Service SHALL use deterministic profiling only.
3. WHEN the Column_Grounder or Predicate_Grounder LLM fallback is unavailable, THE grounder SHALL route to Clarification_Service using deterministic candidates only.
4. WHEN the Semantic_Repair LLM is unavailable, THE Semantic_Pipeline SHALL route to Clarification_Service with the original structural gap rather than retrying or guessing.
5. WHEN LLM coverage Shadow_Mode experiences a failure, THE failure SHALL have no production impact; the pipeline SHALL continue without the shadow comparison metric.
6. THE Semantic_Pipeline SHALL distinguish interpretation_failed (system/provider error requiring operational recovery) from needs_clarification (genuine user ambiguity requiring a specific clarification question).

### Requirement 19: Feature-Flag Compatibility Constraints

**User Story:** As a deployment engineer, I want invalid feature-flag combinations to be caught at startup, so that the system cannot enter a mixed mode where both legacy and new grounders own the same decision.

#### Acceptance Criteria

1. THE Semantic_Pipeline SHALL define legal feature-flag combinations as a compatibility matrix validated at application startup.
2. IF ENABLE_PREFLIGHT_GROUNDING is true, THEN DISABLE_EXECUTION_TIME_GROUNDING SHALL also be true; violation SHALL cause startup failure.
3. IF ENABLE_SEMANTIC_DRAFT_PIPELINE is false, THEN ENABLE_CLARIFICATION_AS_DRAFT_PATCHING SHALL also be false; violation SHALL cause startup failure.
4. IF ENABLE_LLM_COVERAGE_SHADOW is true, THEN ENABLE_DETERMINISTIC_COVERAGE SHALL also be true; violation SHALL cause startup failure.
5. WHEN an invalid flag combination is detected at startup, THE Semantic_Pipeline SHALL log the conflicting flags, the expected valid state, and terminate with a non-zero exit code.

### Requirement 20: Resolution Status and Origin Separation

**User Story:** As a developer, I want resolution validity separated from resolution origin, so that compiler acceptance is a simple type check without encoding workflow history in a validity field.

#### Acceptance Criteria

1. THE SemanticIntentDraft SHALL include a resolution_status field with allowed values: pending, needs_clarification, interpretation_failed, unsupported, invalid, or resolved.
2. THE CanonicalIntent SHALL be a distinct type that can only be constructed when resolution_status is "resolved"; the type-level guarantee replaces runtime status checks.
3. THE CanonicalIntent SHALL include a resolution_origin field with allowed values: direct, semantic_repair, automatic_grounding, or user_clarification.
4. THE Compiler SHALL accept only CanonicalIntent objects, which by construction are always resolved regardless of resolution_origin.
5. THE Semantic_Pipeline SHALL set resolution_origin to reflect the workflow path that produced the final resolved state, independent of resolution_status transitions.

### Requirement 21: Canonicalization and Finalization Boundary

**User Story:** As a pipeline architect, I want a clear type-level boundary between draft (mutable, possibly unresolved) and canonical (immutable, always resolved), so that no partially-resolved structure can be misused as executable input.

#### Acceptance Criteria

1. THE SemanticIntentDraft MAY use any Resolution_Status including pending, needs_clarification, interpretation_failed, unsupported, invalid, or resolved.
2. THE CanonicalIntent SHALL only represent a fully resolved executable meaning; it SHALL be constructable only from a SemanticIntentDraft with resolution_status "resolved" after all grounding and clarification completes.
3. THE Intent_Resolution_Coordinator SHALL be the final owner of operation classification and boolean scope; it SHALL finalize action type and logical-group structure before grounding dispatch.
4. WHEN canonicalization occurs, THE Semantic_Pipeline SHALL verify that semantic validation, required clarification, and grounding are all complete before constructing the CanonicalIntent.
5. AFTER canonicalization, THE semantic action type and logical grouping within the CanonicalIntent SHALL be immutable; later grounding or execution stages SHALL NOT alter canonical semantic structure.
