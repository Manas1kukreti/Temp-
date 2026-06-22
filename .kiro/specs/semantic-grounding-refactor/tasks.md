# Implementation Plan: Semantic Grounding Refactor

## Overview

This plan implements the refactoring of FinFlow's natural-language interpretation and grounding architecture into a multi-stage pipeline with separated responsibilities. The implementation proceeds in layers: shared data models first, then pipeline infrastructure, then individual stage components, then integration wiring. Python with Pydantic models, Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up package structure and shared data models
  - [x] 1.1 Create directory structure and package init files
    - Create `grounding/`, `pipeline/`, `models/` packages under `finflow_agent/`
    - Add `__init__.py` files with appropriate exports
    - _Requirements: Architecture layout from design_

  - [x] 1.2 Implement provenance models (`models/provenance.py`)
    - Define `PromptSpanProvenance`, `ClarificationProvenance`, `SchemaEvidenceProvenance`
    - Define `ProvenanceRef` discriminated union (discriminator="type")
    - Use `ConfigDict(strict=True)` to prevent silent coercion
    - _Requirements: 1.6, 15.1, 15.5, 14.6_

  - [x] 1.3 Implement SemanticIntentDraft model (`models/draft.py`)
    - Define `ReferenceKind`, `ResolutionStatus`, `ResolutionOrigin` enums
    - Define `SemanticColumnReference` with reference_kind classification and provenance
    - Define `AmbiguityMarker`, `LogicalGroup`, `UnresolvedPredicate`
    - Define discriminated action union: `FilterAction`, `ProjectAction`, `DropAction`, `SortAction`, `RenameAction`, `DraftAction`
    - Define `SemanticIntentDraft` with schema_version, draft_revision, resolution_status, resolution_origin, and all fields
    - _Requirements: 1.1, 1.2, 1.5, 14.1, 14.2, 14.4, 14.5, 14.6, 17.1, 20.1_

  - [x] 1.4 Implement CanonicalIntent model (`models/canonical.py`)
    - Define `CanonicalIntent` with `frozen=True`, `resolution_status` literal "resolved"
    - Implement `from_resolved_draft()` factory that validates resolution_status and all references resolved
    - Raise `CanonicalizeError` for unresolved drafts
    - _Requirements: 6.2, 11.2, 20.2, 20.4, 21.1, 21.2, 21.4, 21.5_

  - [x] 1.5 Implement fingerprint and snapshot models (`models/fingerprints.py`, `models/snapshot.py`)
    - Define `StructuralSchemaFingerprint` with deterministic `compute()` method (SHA-256)
    - Define `ProfileFingerprint` with cardinality buckets and representative value hashes (no raw values)
    - Define `DataSnapshotRef` with file_id, content_hash, byte_size, storage_version, profile_id, fingerprints
    - _Requirements: 5.1, 5.2, 5.6, 16.3_

  - [x] 1.6 Implement patch and envelope models (`models/patches.py`, `models/envelope.py`)
    - Define `PatchOp` enum (add, replace, remove) and `SemanticPatch` model
    - Define `PipelineStatus` enum and `IntentEnvelope` container
    - Define `ResolutionRecord` and `ShadowComparisonMetric`
    - _Requirements: 4.3, 4.4, 20.1_

  - [x]* 1.7 Write property test: SemanticIntentDraft serialization round-trip
    - **Property 1: SemanticIntentDraft Serialization Round-Trip**
    - Generate arbitrary valid drafts with all action variants, provenance types, resolution statuses
    - Verify `model_validate(model_dump(mode="json"))` produces equivalent object
    - **Validates: Requirements 14.1, 14.2, 14.3, 14.5**

  - [x]* 1.8 Write property test: Canonicalization type-level guarantee
    - **Property 11: Canonicalization Type-Level Guarantee**
    - Generate drafts with various resolution statuses and unresolved references
    - Verify only fully-resolved drafts with all columns grounded produce CanonicalIntent
    - Verify unresolved drafts raise CanonicalizeError
    - **Validates: Requirements 6.2, 11.2, 20.2, 20.4, 21.2, 21.4**

  - [x]* 1.9 Write property test: Canonical Intent immutability
    - **Property 18: Canonical Intent Immutability**
    - Generate valid CanonicalIntent objects
    - Verify any mutation attempt raises an error (frozen model)
    - **Validates: Requirements 21.5**

  - [x]* 1.10 Write property test: Draft schema version rejection
    - **Property 19: Draft Schema Version Rejection**
    - Generate draft JSON payloads with schema_version > current supported
    - Verify pipeline rejects with version-incompatibility error
    - **Validates: Requirements 14.4**

- [x] 2. Implement pipeline infrastructure
  - [x] 2.1 Implement feature flag system (`pipeline/feature_flags.py`)
    - Define `FeatureFlags` model with all flag fields and defaults
    - Implement `COMPATIBILITY_RULES` as class variable
    - Implement `validate_compatibility()` returning list of errors
    - Startup validation that terminates with non-zero exit on invalid combinations
    - Log conflicting flags and expected valid state on failure
    - _Requirements: 13.1, 13.3, 13.6, 19.1, 19.2, 19.3, 19.4, 19.5_

  - [x]* 2.2 Write property test: Feature-flag compatibility validation
    - **Property 14: Feature-Flag Compatibility Validation**
    - Generate all 2^8 flag combinations
    - Verify invalid combinations are caught (preflight without disabling exec-time; draft patching without draft pipeline; LLM shadow without deterministic coverage)
    - Verify valid combinations pass
    - **Validates: Requirements 13.3, 13.6, 19.1, 19.2, 19.3, 19.4, 19.5**

  - [x] 2.3 Implement observability module (`pipeline/observability.py`)
    - Define structured tracing context with required fields (submission_id, draft_id, draft_revision, intent_id, schema_fingerprint, profile_fingerprint, data_snapshot_ref, model_version, pipeline_stage, decision_owner, duration_ms)
    - Define metric emitters for all pipeline events (extraction, grounding, repair, coverage, clarification)
    - Shadow mode comparison metric recording
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 2.4 Implement LLM adapter protocol (`grounding/llm_adapter.py`)
    - Define `LLMCallSite` enum (extraction, repair, schema_inference, column_grounding, predicate_grounding, coverage_shadow)
    - Define `LLMConstraint` model (output schema, allowed operations, retry policy)
    - Define `SemanticResolver` protocol with `call()` method
    - Define `RetryPolicy` model with bounded retries and backoff
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 18.1_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement coverage validation and bounded repair
  - [x] 4.1 Implement Coverage Validator (`pipeline/coverage_validator.py`)
    - Structural checks: JSON/schema validity, action-to-reference completeness, negation/boolean-group preservation, duplicate/contradictory action detection, unresolved-reference declarations, valid contract tokens
    - Provenance completeness: every element has ProvenanceRef, every material span linked
    - Shadow LLM mode (behind feature flag, no authority)
    - Emit `ShadowComparisonMetric` when shadow active
    - No keyword analysis or semantic re-interpretation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x]* 4.2 Write property test: Provenance completeness invariant
    - **Property 2: Provenance Completeness Invariant**
    - Generate valid SemanticIntentDraft objects
    - Verify every semantic element has at least one ProvenanceRef
    - Verify every material source span is linked
    - **Validates: Requirements 1.6, 3.2, 15.1, 15.2, 15.3**

  - [x] 4.3 Implement Semantic Repair (`pipeline/semantic_repair.py`)
    - Accept only declared patch paths from validator failures
    - Maximum one attempt per pipeline invocation
    - Return typed SemanticPatch list (add/replace/remove only)
    - Log patch path, input failure, operation type, resulting modification
    - _Requirements: 4.1, 4.2, 4.3, 4.6_

  - [x] 4.4 Implement patch application logic
    - Deterministic application of SemanticPatch list to draft
    - Produce new draft revision (N → N+1), original immutable
    - Include ProvenanceRef for new/modified elements
    - _Requirements: 4.4, 15.4, 17.1_

  - [x]* 4.5 Write property test: Bounded repair constraint
    - **Property 5: Bounded Repair Constraint**
    - Generate pipeline invocations with structural gaps
    - Verify at most one repair attempt, only typed patches returned
    - Verify gaps after repair → needs_clarification
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.5**

  - [x]* 4.6 Write property test: Patch application produces new revision
    - **Property 6: Patch Application Produces New Revision**
    - Generate random drafts at revision N and valid patch sets
    - Verify result is revision N+1, original draft unchanged
    - **Validates: Requirements 4.4, 17.1**

- [x] 5. Implement schema service and preflight loading
  - [x] 5.1 Implement Preflight Data Loader (`grounding/preflight_loader.py`)
    - Load source file and produce DataFrameProfile + DataSnapshotRef
    - Read-only mode: no cleaning, transformation, or mutation
    - Compute content_hash as part of DataSnapshotRef
    - Enforce configurable size limits (reject oversized files)
    - _Requirements: 16.1, 16.2, 16.3, 16.5, 16.6_

  - [x] 5.2 Implement Schema Service with layered cache (`grounding/schema_service.py`)
    - L1 structural role cache keyed by StructuralSchemaFingerprint + role-model version
    - L2 value-evidence cache keyed by structural fingerprint + ProfileFingerprint + profiler version
    - Return cached result on fingerprint match without re-computation
    - Execute during dataset-profiling stage (after preflight, before grounding)
    - Graceful degradation: return cached or deterministic-only on LLM failure
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 18.2_

  - [x]* 5.3 Write property test: Schema cache determinism
    - **Property 7: Schema Cache Determinism**
    - Generate schema inference requests with identical/different fingerprints
    - Verify identical inputs → identical results
    - Verify different structural schemas → different fingerprints
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 8.3**

  - [x]* 5.4 Write property test: Content hash consistency enforcement
    - **Property 17: Content Hash Consistency Enforcement**
    - Generate execution scenarios with matching/mismatching content hashes
    - Verify fail-closed behavior on mismatch
    - **Validates: Requirements 16.3, 16.4**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement shared candidate generation and grounding
  - [x] 7.1 Implement Candidate Generation Layer (`grounding/candidate_generator.py`)
    - Shared interface for both Column Grounder and Predicate Grounder
    - Deterministic scoring: token overlap, value-concept matching, semantic-type alignment, column-name similarity
    - Expose positive + negative evidence per candidate
    - Identical inputs → identical scores guarantee
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 7.2 Implement scoring utilities (`grounding/scoring.py`)
    - Token overlap scoring
    - Value-concept matching
    - Semantic-type alignment scoring
    - Column-name similarity (normalized)
    - _Requirements: 7.2_

  - [x] 7.3 Implement evidence models (`grounding/evidence.py`)
    - Define `ScoredCandidate` with score breakdown and evidence lists
    - Define `GroundingConfig` (confidence_threshold, ambiguity_margin, llm_fallback_enabled, destructive_action_extra_caution)
    - _Requirements: 7.4, 8.9_

  - [x]* 7.4 Write property test: Candidate scoring determinism
    - **Property 8: Candidate Scoring Determinism**
    - Generate random references and profiles
    - Verify same input → same output across multiple calls, regardless of call order
    - **Validates: Requirements 7.2, 7.3, 7.4**

  - [x] 7.5 Implement post-LLM verification (`grounding/verification.py`)
    - Define `PostLLMVerification` model with all checks (candidate_exists, operator_compatible, value_shape_valid, dtype_compatible, in_permitted_set, tie_breaking_result)
    - Implement `apply_tie_breaking_policy()` function with full ruleset
    - Route to Clarification on any verification failure
    - _Requirements: 8.7, 8.8, 8.9_

  - [x]* 7.6 Write property test: Tie-breaking policy correctness
    - **Property 9: Tie-Breaking Policy Correctness**
    - Generate all combinations of scores, margins, destructive flags
    - Verify RESOLVE only when LLM agrees with strong leader and not destructive-close
    - Verify CLARIFY for close ties, conflicts, and destructive operations
    - **Validates: Requirements 8.9**

  - [x]* 7.7 Write property test: Post-LLM verification gate
    - **Property 10: Post-LLM Verification Gate**
    - Generate LLM fallback selections with various verification outcomes
    - Verify all checks applied; failure → Clarification (not acceptance)
    - **Validates: Requirements 8.7, 8.8**

  - [x] 7.8 Implement Column Grounder (`grounding/column_grounder.py`)
    - Resolve standalone column references (projections, sorts, drops, renames)
    - Use Candidate Generation Layer for scoring
    - LLM fallback constrained to existing physical columns
    - Post-LLM verification before accepting
    - Does NOT resolve filter predicate columns
    - _Requirements: 2.4, 8.4_

  - [x] 7.9 Implement Predicate Grounder (`grounding/predicate_grounder.py`)
    - Resolve complete filter predicates (field + operator + value)
    - Own filter column resolution, operator mapping, value normalization
    - Does NOT delegate filter-column decisions to Column Grounder
    - Use Candidate Generation Layer for column scoring
    - LLM fallback constrained to existing physical columns
    - Post-LLM verification before accepting
    - _Requirements: 2.5, 8.5_

  - [x]* 7.10 Write property test: Single decision-owner routing
    - **Property 3: Single Decision-Owner Routing**
    - Generate references with known context (standalone vs filter predicate)
    - Verify routing to exactly one grounder, rejection if presented to wrong one
    - **Validates: Requirements 2.3, 2.4, 2.5, 8.4, 8.5**

  - [x]* 7.11 Write property test: Finalized resolution immutability
    - **Property 4: Finalized Resolution Immutability**
    - Generate semantic elements with finalized resolutions
    - Verify override attempts by different components are rejected
    - **Validates: Requirements 2.6**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement pipeline coordination and clarification
  - [x] 9.1 Implement Intent Resolution Coordinator (`pipeline/coordinator.py`)
    - Finalize operation classification and boolean scope
    - Receive operation candidates from extraction
    - Apply deterministic validation results
    - Apply user clarification patches
    - Select unique operation when supported, preserve ambiguity otherwise
    - Return updated draft with finalized action type and logical-group structure
    - _Requirements: 2.1, 2.2, 21.3_

  - [x] 9.2 Implement Clarification Service draft-patching extension
    - Patch existing SemanticIntentDraft with user selection (new immutable revision)
    - Record ClarificationProvenance (question_id, response_id, selected_value)
    - Stage-resume policy: column → grounding; operation → validation + grounding; value → predicate grounding; prompt replacement → extraction
    - Stale revision rejection (expected_revision check)
    - Idempotency key enforcement
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 17.2, 17.3, 17.4_

  - [x]* 9.3 Write property test: Stale revision rejection
    - **Property 15: Stale Revision Rejection**
    - Generate patch requests with various revision numbers
    - Verify rejection when expected_revision differs from current
    - Verify duplicate idempotency_key → rejected or no-op
    - **Validates: Requirements 17.2, 17.3, 17.4**

  - [x]* 9.4 Write property test: Clarification produces ClarificationProvenance
    - **Property 20: Clarification Produces ClarificationProvenance**
    - Generate user clarification responses that patch drafts
    - Verify resulting revision contains ClarificationProvenance (not synthetic PromptSpanProvenance)
    - **Validates: Requirements 15.5, 9.2**

  - [x]* 9.5 Write property test: Uncertainty resolution policy
    - **Property 13: Uncertainty Resolution Policy**
    - Generate grounding results with various confidence levels and ambiguity margins
    - Verify route to Clarification when below threshold or within margin
    - Verify low-confidence never proceeds to execution with only a warning
    - **Validates: Requirements 6.5, 10.2, 10.3, 10.4, 10.6**

  - [x]* 9.6 Write property test: LLM failure classification
    - **Property 16: LLM Failure Classification**
    - Generate LLM provider failures (timeout, rate limit, invalid JSON, schema validation failure, empty output)
    - Verify extraction failures → interpretation_failed (never needs_clarification)
    - **Validates: Requirements 18.1, 18.6**

- [x] 10. Implement execution boundary and canonicalization
  - [x] 10.1 Implement Canonicalizer (`pipeline/canonicalizer.py`)
    - Verify semantic validation, clarification, and grounding are all complete
    - Construct CanonicalIntent only from fully-resolved draft
    - Verify no unresolved active execution references remain
    - Type-level boundary enforcement
    - _Requirements: 21.2, 21.4, 6.2_

  - [x] 10.2 Implement Compiler contract (`planning/compiler.py` modification)
    - Accept only CanonicalIntent objects (type-level guarantee)
    - Validate every column reference resolved to physical column
    - No grounding, no LLM calls, no semantic re-interpretation
    - Produce ExecutionPlan
    - _Requirements: 6.1, 6.4, 11.1, 11.2_

  - [x] 10.3 Implement Executor contract (`execution/engine.py` modification)
    - Validate every referenced column exists in intent package
    - Verify content_hash matches DataSnapshotRef
    - Fail-closed on column not in validated package
    - Zero LLM calls, zero grounding, zero semantic interpretation
    - _Requirements: 6.3, 8.6, 11.3, 11.4, 11.5, 16.4_

  - [x]* 10.4 Write property test: Executor boundary contract
    - **Property 12: Executor Boundary Contract**
    - Generate ExecutionPlan steps referencing columns not in intent package
    - Verify fail-closed error without executing
    - Verify zero LLM calls during execution
    - **Validates: Requirements 6.3, 8.6, 11.3, 11.4, 11.5**

- [x] 11. Implement migration and legacy support
  - [x] 11.1 Implement versioned upcasters (`models/upcasters.py`)
    - Convert legacy CanonicalIntent objects to new schema format
    - Preserve all semantic meaning from original payload
    - Version detection and appropriate upcaster selection
    - _Requirements: 13.4_

  - [x] 11.2 Implement legacy feature-flag routing
    - When flags disabled, route to legacy behavior
    - Maintain backward compatibility during migration
    - Log flag transitions with previous state, new state, timestamp
    - _Requirements: 13.1, 13.2, 13.5_

  - [x]* 11.3 Write property test: Legacy upcaster round-trip
    - **Property 21: Legacy Upcaster Round-Trip**
    - Generate valid legacy CanonicalIntent payloads
    - Verify upcaster produces valid new-schema CanonicalIntent preserving semantic meaning
    - **Validates: Requirements 13.4**

- [x] 12. Implement Semantic Extractor integration
  - [x] 12.1 Implement Semantic Extractor (`grounding/semantic_extractor.py` or existing integration point)
    - Convert raw prompt to SemanticIntentDraft (not CanonicalIntent)
    - Preserve ambiguity: multiple interpretations as ambiguity markers
    - Classify generic words as generic_reference
    - Preserve boolean scope (value sets not split)
    - Include typed ProvenanceRef for every extracted element
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 8.1, 15.1_

  - [x] 12.2 Implement resolution status and origin handling
    - Separation of resolution_status (validity) from resolution_origin (workflow path)
    - Set resolution_origin to reflect actual workflow path
    - _Requirements: 20.1, 20.3, 20.5_

- [x] 13. Integration wiring: Assemble full pipeline
  - [x] 13.1 Wire pipeline orchestrator
    - Connect all stages in order: Extractor → Coverage Validator → Repair → Coordinator → Preflight → Schema → Candidate Gen → Column Grounder + Predicate Grounder → Canonicalizer → Compiler → Executor
    - Implement stage-resume on clarification
    - Apply resolution policies for each uncertainty class
    - Feature-flag-aware routing (legacy vs new paths)
    - _Requirements: 6.1, 6.4, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 13.2 Implement SemanticIntentDraft pretty-printer
    - Format draft objects into human-readable structured text
    - Support debugging and logging use cases
    - _Requirements: 14.7_

  - [x]* 13.3 Write integration tests for end-to-end pipeline
    - Test prompt → CanonicalIntent → ExecutionPlan → result
    - Test feature flag toggle (legacy → new behavior)
    - Test clarification flow (ambiguity → user response → resume)
    - Test LLM failure degradation paths
    - _Requirements: All_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (21 properties from the design)
- Unit tests validate specific examples and edge cases
- The design uses Python with Pydantic models and Hypothesis for property-based testing
- All LLM call sites have bounded behavior and explicit constraints
- The pipeline is deployed incrementally behind feature flags with compatibility validation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.5"] },
    { "id": 2, "tasks": ["1.3", "1.6"] },
    { "id": 3, "tasks": ["1.4", "1.7", "1.10"] },
    { "id": 4, "tasks": ["1.8", "1.9", "2.1", "2.3", "2.4"] },
    { "id": 5, "tasks": ["2.2", "4.1", "5.1"] },
    { "id": 6, "tasks": ["4.2", "4.3", "4.4", "5.2"] },
    { "id": 7, "tasks": ["4.5", "4.6", "5.3", "5.4"] },
    { "id": 8, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 9, "tasks": ["7.4", "7.5"] },
    { "id": 10, "tasks": ["7.6", "7.7", "7.8", "7.9"] },
    { "id": 11, "tasks": ["7.10", "7.11", "9.1", "9.2"] },
    { "id": 12, "tasks": ["9.3", "9.4", "9.5", "9.6"] },
    { "id": 13, "tasks": ["10.1", "12.1", "12.2"] },
    { "id": 14, "tasks": ["10.2", "10.3"] },
    { "id": 15, "tasks": ["10.4", "11.1", "11.2"] },
    { "id": 16, "tasks": ["11.3", "13.1", "13.2"] },
    { "id": 17, "tasks": ["13.3"] }
  ]
}
```
