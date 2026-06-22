# Implementation Plan: Compiler-Executor Contract

## Overview

Introduce a shared contract registry module (`src/finflow_agent/contract_registry.py`) that defines Python Enum classes for all canonical operators and action kinds, co-locates semantic-to-canonical mappings, provides validation functions, and enforces coverage checks at import time. Integrate the registry into the semantic compiler (output validation), execution engine (handler coverage), and schema layer (typed operator field). All code is Python, tested with Hypothesis for property-based tests and pytest for unit/integration tests.

## Tasks

- [x] 1. Create contract registry module with enums and exceptions
  - [x] 1.1 Create `src/finflow_agent/contract_registry.py` with `CanonicalOperator(str, Enum)` containing all 15 operators, `ActionKind(str, Enum)` containing all 10 action kinds, the exception hierarchy (`ContractViolationError`, `InvalidOperatorError`, `InvalidActionKindError`, `UnmappedSemanticTypeError`), and the two semantic mapping dicts (`SEMANTIC_RELATION_TO_OPERATOR`, `SEMANTIC_OPERATION_TO_ACTION_KIND`)
    - Define `CanonicalOperator` with members: EQ, NEQ, GT, GTE, LT, LTE, CONTAINS, NOT_CONTAINS, STARTS_WITH, ENDS_WITH, BETWEEN, IN, NOT_IN, IS_NULL, IS_NOT_NULL
    - Define `ActionKind` with members: CLEAN, PROJECT_COLUMNS, DROP_COLUMNS, RENAME_COLUMNS, FILTER_ROWS, SORT_ROWS, LIMIT_ROWS, CALCULATE, VISUALIZE, REPORT
    - Ensure str mixin pattern so `CanonicalOperator.EQ == "eq"` is True
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 5.2, 5.4, 8.1, 8.2_

  - [x] 1.2 Implement validation functions: `validate_operator`, `validate_action_kind`, `resolve_semantic_operator`, `resolve_semantic_operation_type`
    - `validate_operator(s)` returns `CanonicalOperator` or raises `InvalidOperatorError` with invalid value and valid list
    - `validate_action_kind(s)` returns `ActionKind` or raises `InvalidActionKindError` with invalid value and valid list
    - `resolve_semantic_operator(r)` maps semantic relation to canonical or raises `UnmappedSemanticTypeError` with suggestion
    - `resolve_semantic_operation_type(t)` maps semantic op type to canonical or raises `UnmappedSemanticTypeError` with suggestion
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.3, 9.2, 9.3, 9.4_

  - [x] 1.3 Implement coverage check functions: `check_operator_handler_coverage`, `check_action_kind_coverage`
    - Both perform bidirectional validation (missing handlers AND unknown/excess handlers)
    - Raise `ImportError` identifying specific unhandled or unknown members
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 9.1_

  - [ ]* 1.4 Write unit tests for contract registry module
    - Verify `CanonicalOperator` contains exactly 15 members with correct string values
    - Verify `ActionKind` contains exactly 10 members with correct string values
    - Test `validate_operator` with valid and invalid inputs
    - Test `validate_action_kind` with valid and invalid inputs
    - Test `resolve_semantic_operator` and `resolve_semantic_operation_type` for all mapped keys
    - Test coverage check functions with complete, incomplete, and excess registries
    - Test error messages contain expected diagnostic information
    - _Requirements: 1.4, 2.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4_

- [x] 2. Integrate contract registry into semantic compiler
  - [x] 2.1 Modify `src/finflow_agent/planning/compiler.py` to import and use contract registry validation
    - Import `validate_operator`, `validate_action_kind`, `InvalidOperatorError`, `InvalidActionKindError` from `contract_registry`
    - Add `_validated_operator(operator: str) -> str` wrapper that catches `InvalidOperatorError` and raises `SemanticCompilationError`
    - Add `_validated_action_kind(kind: str) -> str` wrapper that catches `InvalidActionKindError` and raises `SemanticCompilationError`
    - Call validation on every operator token and action kind before emitting compiled output
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 2.2 Replace hardcoded semantic-to-canonical mappings in compiler with `resolve_semantic_operator` and `resolve_semantic_operation_type` calls
    - Remove any inline `OPERATOR_MAP` or equivalent dicts from compiler
    - Use `resolve_semantic_operator` for filter operator resolution
    - Use `resolve_semantic_operation_type` for action kind resolution
    - Wrap `UnmappedSemanticTypeError` as `SemanticCompilationError`
    - _Requirements: 5.2, 5.3, 6.1, 6.3, 7.3, 7.4_

  - [ ]* 2.3 Write property test: operator validation accepts/rejects correctly (Property 1)
    - **Property 1: Operator validation accepts all valid and rejects all invalid tokens**
    - Use Hypothesis `st.text()` to generate arbitrary strings
    - Assert: if string is a valid operator value → returns enum member; otherwise → raises `InvalidOperatorError`
    - Apply symmetrically to `validate_action_kind`
    - **Validates: Requirements 3.1, 3.2**

  - [ ]* 2.4 Write property test: invalid operator errors are descriptive (Property 2)
    - **Property 2: Invalid operator errors are descriptive**
    - Generate strings NOT in the valid operator set
    - Assert error message contains both the invalid value and complete list of valid operators
    - **Validates: Requirements 3.3, 3.4, 9.2, 9.3**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Integrate contract registry into execution engine
  - [x] 4.1 Modify `src/finflow_agent/operations/filter_handlers.py` to call `check_operator_handler_coverage` at module level
    - Import `check_operator_handler_coverage` from `contract_registry`
    - Add call after `FILTER_HANDLERS` dict is defined
    - Ensure all 15 operators have handler entries (add placeholder/pass-through handlers for any missing)
    - _Requirements: 4.1, 4.3_

  - [x] 4.2 Modify `src/finflow_agent/operations/executor.py` to create `ACTION_HANDLERS` registry and call `check_action_kind_coverage` at module level
    - Import `check_action_kind_coverage` from `contract_registry`
    - Define `ACTION_HANDLERS` dict mapping action kind strings to handler functions
    - Add coverage check call after dict definition
    - Ensure all 10 action kinds have handler entries
    - _Requirements: 4.2, 4.4, 6.2_

  - [ ]* 4.3 Write property test: semantic-to-canonical operator mapping coverage (Property 3)
    - **Property 3: Semantic-to-canonical operator mapping coverage**
    - For each key in `SEMANTIC_RELATION_TO_OPERATOR`, assert `resolve_semantic_operator(key)` returns a valid `CanonicalOperator` member
    - **Validates: Requirements 7.3**

  - [ ]* 4.4 Write property test: semantic-to-canonical action kind mapping coverage (Property 4)
    - **Property 4: Semantic-to-canonical action kind mapping coverage**
    - For each key in `SEMANTIC_OPERATION_TO_ACTION_KIND`, assert `resolve_semantic_operation_type(key)` returns a valid `ActionKind` member
    - **Validates: Requirements 7.4, 6.1**

  - [ ]* 4.5 Write property test: unmapped semantic types produce descriptive errors (Property 5)
    - **Property 5: Unmapped semantic types produce descriptive errors**
    - Generate strings NOT in `SEMANTIC_OPERATION_TO_ACTION_KIND` keys
    - Assert `UnmappedSemanticTypeError` is raised containing the value and "registration" suggestion text
    - **Validates: Requirements 6.3, 9.4**

- [x] 5. Integrate contract registry into schema layer
  - [x] 5.1 Update `FilterCondition` model in `src/finflow_agent/operations/schemas.py` to use `CanonicalOperator` enum for the `operator` field
    - Import `CanonicalOperator` from `contract_registry`
    - Change `operator` field type from `Literal[...]` to `CanonicalOperator`
    - Verify Pydantic accepts both raw strings and enum members (backward compat)
    - _Requirements: 8.3, 8.4_

  - [x] 5.2 Update `FilterCondition` in `src/finflow_agent/planning/canonical_intent.py` to use `CanonicalOperator` enum for the `operator` field
    - Import `CanonicalOperator` from `contract_registry`
    - Change `operator` field type from `Literal[...]` to `CanonicalOperator`
    - _Requirements: 8.3, 8.4_

  - [ ]* 5.3 Write property test: enum serialization round-trip (Property 7)
    - **Property 7: Enum serialization round-trip**
    - For each `CanonicalOperator` member, serialize a Pydantic model to JSON and deserialize back
    - Assert the deserialized value equals the original enum member
    - Repeat for `ActionKind`
    - **Validates: Requirements 8.3, 8.4**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Write handler correctness property test and integration tests
  - [ ]* 7.1 Write property test: all registered operators produce valid boolean masks (Property 6)
    - **Property 6: All registered operators produce valid boolean masks**
    - Use Hypothesis pandas strategies to generate compatible Series
    - For each operator in `CanonicalOperator`, invoke the corresponding handler
    - Assert output is a `pd.Series` of boolean dtype with length equal to input Series length
    - **Validates: Requirements 7.1**

  - [ ]* 7.2 Write integration tests for end-to-end contract enforcement
    - Test: compile a canonical intent end-to-end, verify no contract violations
    - Test: load stored canonical intents from JSON fixtures, verify deserialization succeeds with enum members
    - Test: simulate adding an enum member without a handler, verify `ImportError` at import time
    - _Requirements: 7.1, 7.2, 8.3_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The contract registry module has zero internal dependencies (only stdlib + Pydantic) to avoid circular imports
- The `str, Enum` mixin pattern ensures backward compatibility with stored canonical intents — no migration needed
- Bidirectional coverage checks prevent both missing handlers AND stale/unknown handlers from accumulating

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4", "2.1", "5.1", "5.2"] },
    { "id": 3, "tasks": ["2.2", "4.1", "4.2"] },
    { "id": 4, "tasks": ["2.3", "2.4", "4.3", "4.4", "4.5", "5.3"] },
    { "id": 5, "tasks": ["7.1", "7.2"] }
  ]
}
```
