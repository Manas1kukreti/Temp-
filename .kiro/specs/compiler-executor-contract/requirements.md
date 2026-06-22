# Requirements Document

## Introduction

The FinFlow system's semantic compiler translates LLM-extracted semantic operations into canonical actions, and the execution engine runs those canonical actions against dataframes. Currently these two subsystems are connected by implicit string contracts — the compiler emits operator strings and action kind strings that the executor must independently know how to handle. This implicit coupling has caused silent-failure bugs (e.g., the `in_` semantic operator mapped to `eq` instead of `in`, producing wrong results with no error).

This feature introduces a shared contract layer — a single source-of-truth registry that both compiler and executor import — making it structurally impossible for the compiler to produce an operator or action kind that the executor cannot handle.

## Glossary

- **Semantic_Compiler**: The deterministic module (`backend/app/services/semantic_compiler.py`) that translates grounded semantic operations into canonical intent actions.
- **Execution_Engine**: The agent-framework module (`finflow_agent/operations/executor.py` and handler modules) that applies canonical actions to pandas DataFrames.
- **Contract_Registry**: The proposed shared module that defines all valid canonical operators, action kinds, and resolution methods as the single source of truth.
- **Canonical_Operator**: A string token (e.g., `eq`, `in`, `not_in`) that identifies a filter comparison operation in the canonical intent.
- **Action_Kind**: A string token (e.g., `filter_rows`, `clean`, `project_columns`) that identifies the type of canonical action to execute.
- **Handler**: A function registered in the Execution_Engine that implements the behavior for a specific Canonical_Operator or Action_Kind.
- **Operator_Enum**: A Python enum class within the Contract_Registry that enumerates all valid Canonical_Operators.
- **Action_Kind_Enum**: A Python enum class within the Contract_Registry that enumerates all valid Action_Kinds.
- **Coverage_Check**: An import-time or startup-time validation that verifies every member of an enum has a corresponding Handler registered.
- **Canonical_Intent**: The stored JSON representation of compiled actions, persisted in the database for replay and audit.

## Requirements

### Requirement 1: Shared Canonical Operator Enum

**User Story:** As a developer, I want all valid canonical filter operators defined in a single shared enum, so that both the compiler and executor reference the same set of operators without duplication.

#### Acceptance Criteria

1. THE Contract_Registry SHALL define an Operator_Enum containing all valid Canonical_Operators as enum members.
2. THE Operator_Enum SHALL be the sole definition of valid Canonical_Operators across both the Semantic_Compiler and Execution_Engine packages.
3. WHEN a new Canonical_Operator is added to the Operator_Enum, THE Contract_Registry SHALL require no other file to redefine that operator string.
4. THE Operator_Enum SHALL include at minimum the operators: eq, neq, gt, gte, lt, lte, contains, not_contains, starts_with, ends_with, between, in, not_in, is_null, is_not_null.

### Requirement 2: Shared Action Kind Enum

**User Story:** As a developer, I want all valid canonical action kinds defined in a single shared enum, so that the compiler can only emit action kinds the executor knows about.

#### Acceptance Criteria

1. THE Contract_Registry SHALL define an Action_Kind_Enum containing all valid Action_Kinds as enum members.
2. THE Action_Kind_Enum SHALL be the sole definition of valid Action_Kinds across both the Semantic_Compiler and Execution_Engine packages.
3. THE Action_Kind_Enum SHALL include at minimum the kinds: clean, project_columns, drop_columns, rename_columns, filter_rows, sort_rows, limit_rows, calculate, visualize, report.

### Requirement 3: Compiler Output Validation

**User Story:** As a developer, I want the compiler to validate all emitted operators and action kinds against the shared enums before returning compiled output, so that invalid tokens are caught immediately rather than producing silent failures downstream.

#### Acceptance Criteria

1. WHEN the Semantic_Compiler produces a canonical action, THE Semantic_Compiler SHALL validate the action kind value against the Action_Kind_Enum.
2. WHEN the Semantic_Compiler produces a filter condition, THE Semantic_Compiler SHALL validate the operator value against the Operator_Enum.
3. IF the Semantic_Compiler produces an action kind not present in the Action_Kind_Enum, THEN THE Semantic_Compiler SHALL raise a SemanticCompilationError with a message identifying the invalid action kind.
4. IF the Semantic_Compiler produces an operator not present in the Operator_Enum, THEN THE Semantic_Compiler SHALL raise a SemanticCompilationError with a message identifying the invalid operator.

### Requirement 4: Executor Handler Coverage Validation

**User Story:** As a developer, I want the system to verify at import time that every registered operator and action kind has a corresponding handler, so that missing handlers are caught during development rather than at runtime.

#### Acceptance Criteria

1. WHEN the Execution_Engine module is imported, THE Execution_Engine SHALL verify that every member of the Operator_Enum has a corresponding entry in the filter handler registry.
2. WHEN the Execution_Engine module is imported, THE Execution_Engine SHALL verify that every member of the Action_Kind_Enum has a corresponding execution path.
3. IF a member of the Operator_Enum lacks a corresponding filter handler, THEN THE Execution_Engine SHALL raise an ImportError identifying the unhandled operator.
4. IF a member of the Action_Kind_Enum lacks a corresponding execution path, THEN THE Execution_Engine SHALL raise an ImportError identifying the unhandled action kind.

### Requirement 5: Single-Point Registration for New Operators

**User Story:** As a developer, I want adding a new operator to require registration in one location, so that the compiler mapping and executor handler are co-located and cannot drift apart.

#### Acceptance Criteria

1. WHEN a developer adds a new Canonical_Operator, THE Contract_Registry SHALL provide a registration mechanism that associates the enum member with its semantic-to-canonical mapping metadata.
2. THE Contract_Registry SHALL co-locate the semantic relation operator mapping alongside each Operator_Enum member definition.
3. WHEN a developer adds a new Action_Kind, THE Contract_Registry SHALL provide a registration mechanism that associates the enum member with its semantic operation type mapping.
4. THE Contract_Registry SHALL co-locate the semantic operation type mapping alongside each Action_Kind_Enum member definition.

### Requirement 6: Single-Point Registration for New Action Kinds

**User Story:** As a developer, I want adding a new action kind to require a single registration point that declares the semantic-to-canonical mapping, so that the compiler and executor stay synchronized.

#### Acceptance Criteria

1. WHEN a new Action_Kind is registered in the Contract_Registry, THE Semantic_Compiler SHALL use that registration to determine the canonical action kind for a given SemanticOperationType.
2. WHEN a new Action_Kind is registered in the Contract_Registry, THE Execution_Engine SHALL use that registration to route the action to the correct handler.
3. IF a SemanticOperationType has no mapping in the Contract_Registry, THEN THE Semantic_Compiler SHALL raise a SemanticCompilationError identifying the unmapped operation type.

### Requirement 7: Round-Trip Correctness

**User Story:** As a developer, I want a guarantee that every operator the compiler can produce is handled correctly by the executor, so that no silent data corruption occurs.

#### Acceptance Criteria

1. FOR ALL Canonical_Operators in the Operator_Enum, THE Execution_Engine SHALL produce a valid boolean mask when given a DataFrame column and a filter condition using that operator.
2. FOR ALL Action_Kinds in the Action_Kind_Enum, THE Execution_Engine SHALL produce a valid ExecutionOutput when given a DataFrame and an action of that kind.
3. FOR ALL RelationOperator values in the semantic model, THE Semantic_Compiler SHALL map them to a Canonical_Operator present in the Operator_Enum (round-trip: semantic → canonical → handler exists).
4. FOR ALL SemanticOperationType values that have a mapping, THE Semantic_Compiler SHALL map them to an Action_Kind present in the Action_Kind_Enum (round-trip: semantic → canonical → execution path exists).

### Requirement 8: Backward Compatibility with Stored Canonical Intents

**User Story:** As a developer, I want the new contract system to remain compatible with canonical intents already stored in the database, so that existing jobs can be replayed without modification.

#### Acceptance Criteria

1. THE Operator_Enum member values SHALL use the same string tokens as the current canonical operator strings (eq, neq, gt, gte, lt, lte, contains, not_contains, starts_with, ends_with, between, in, not_in, is_null, is_not_null).
2. THE Action_Kind_Enum member values SHALL use the same string tokens as the current canonical action kind strings (clean, project_columns, drop_columns, rename_columns, filter_rows, sort_rows, limit_rows, calculate, visualize, report).
3. WHEN the Execution_Engine receives a Canonical_Intent from the database containing string-based operator and action kind values, THE Execution_Engine SHALL accept those values if they match an enum member value.
4. THE Contract_Registry SHALL serialize enum members as their string values in JSON output to maintain format compatibility with existing stored intents.

### Requirement 9: Descriptive Error Messages for Contract Violations

**User Story:** As a developer, I want clear error messages when a contract violation occurs, so that debugging mapping issues is fast and obvious.

#### Acceptance Criteria

1. WHEN a Coverage_Check fails at import time, THE error message SHALL identify the specific enum member that lacks a handler.
2. WHEN the Semantic_Compiler rejects an invalid operator, THE error message SHALL include the invalid operator value and the list of valid operators from the Operator_Enum.
3. WHEN the Semantic_Compiler rejects an invalid action kind, THE error message SHALL include the invalid action kind value and the list of valid action kinds from the Action_Kind_Enum.
4. WHEN a SemanticOperationType has no canonical mapping, THE error message SHALL identify the unmapped semantic operation type and suggest adding a registration entry.
