"""Planning package — compilation of intents into execution plans.

Exports both legacy compilation (compile_intent_to_plan) and the refactored
Compiler class from the semantic-grounding-refactor spec.
"""

# Lazy imports to avoid circular dependency chains in the existing codebase.
# Use: from finflow_agent.planning import Compiler
# Or:  from finflow_agent.planning.compiler import Compiler


def __getattr__(name: str):
    """Lazy module-level attribute access for planning package exports."""
    _exports = {
        "CompilerError",
        "ExecutionStep",
        "RefactoredExecutionPlan",
        "Compiler",
    }
    if name in _exports:
        from finflow_agent.planning.compiler import (
            CompilerError,
            ExecutionStep,
            RefactoredExecutionPlan,
            Compiler,
        )
        _mapping = {
            "CompilerError": CompilerError,
            "ExecutionStep": ExecutionStep,
            "RefactoredExecutionPlan": RefactoredExecutionPlan,
            "Compiler": Compiler,
        }
        return _mapping[name]
    raise AttributeError(f"module 'finflow_agent.planning' has no attribute {name!r}")


__all__ = [
    "CompilerError",
    "ExecutionStep",
    "RefactoredExecutionPlan",
    "Compiler",
]
