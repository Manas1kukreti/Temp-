from __future__ import annotations

from app.services.semantic_extractor import parse_llm_semantic_response
from app.services.semantic_pipeline import run_semantic_pipeline_sync


SAMPLE_COLUMNS = [
    "consumer_id",
    "age",
    "gender",
    "income",
    "status",
    "transaction_date",
    "amount",
    "merchant",
    "payment_method",
]


def test_parse_llm_semantic_response_recovers_broken_depends_on_key():
    raw_json = """
    {
      "goals": [
        {
          "description": "Clean this data and return every column except payment mode",
          "priority": 1
        }
      ],
      "tasks": [
        {
          "task_id": "task_1",
          "operation": {
            "type": "clean"
          },
          "inputs": [],
          "parameters": {},
          "depends_on": [],
          "confidence": 0.95
        },
        {
          "task_id": "task_2",
          "operation": {
            "type": "exclude_columns"
          },
          "inputs": [
            {
              "kind": "column_reference",
              "user_term": "payment mode"
            }
          ],
          "parameters": {},
          "depends_on:[\\n            \"task_1\"\\n         ],
          "confidence": 0.95
        }
      ],
      "outputs": [],
      "constraints": [],
      "ambiguities": [],
      "unsupported_requirements": []
    }
    """

    intent = parse_llm_semantic_response(raw_json)

    assert len(intent.tasks) == 2
    assert intent.tasks[1].depends_on == ["task_1"]
    assert intent.tasks[1].operation.type.value == "exclude_columns"


def test_parse_llm_semantic_response_recovers_single_quotes_and_trailing_commas():
    raw_json = """
    ```json
    {
      'goals': [
        {'description': 'Clean this data and return every column except consumer ID', 'priority': 1},
      ],
      'tasks': [
        {
          'task_id': 'task_1',
          'operation': {'type': 'clean'},
          'inputs': [],
          'parameters': {},
          'depends_on': [],
          'confidence': 0.95,
        },
        {
          'task_id': 'task_2',
          'operation': {'type': 'exclude_columns'},
          'inputs': [
            {'kind': 'column_reference', 'user_term': 'consumer ID'},
          ],
          'parameters': {},
          'depends_on': ['task_1'],
          'confidence': 0.95,
        },
      ],
      'outputs': [],
      'constraints': [],
      'ambiguities': [],
      'unsupported_requirements': [],
    }
    ```
    """

    intent = parse_llm_semantic_response(raw_json)

    assert len(intent.goals) == 1
    assert intent.tasks[0].operation.type.value == "clean"
    assert intent.tasks[1].depends_on == ["task_1"]


def test_semantic_pipeline_recovers_and_compiles_malformed_llm_output():
    malformed_json = """
    {
      "goals": [
        {
          "description": "Clean this data and return every column except consumer ID",
          "priority": 1
        }
      ],
      "tasks": [
        {
          "task_id": "task_1",
          "operation": {"type": "clean"},
          "inputs": [],
          "parameters": {},
          "depends_on": [],
          "confidence": 0.95
        },
        {
          "task_id": "task_2",
          "operation": {"type": "exclude_columns"},
          "inputs": [
            {"kind": "column_reference", "user_term": "consumer ID"}
          ],
          "parameters": {},
          "depends_on:[\\n            \"task_1\"\\n         ],
          "confidence": 0.95
        }
      ],
      "outputs": [],
      "constraints": [],
      "ambiguities": [],
      "unsupported_requirements": []
    }
    """

    def mock_llm(messages):
        return malformed_json

    result = run_semantic_pipeline_sync(
        "clean this data and return every column except consumer ID",
        SAMPLE_COLUMNS,
        llm_call=mock_llm,
    )

    assert result.success, result.error
    action_kinds = [action["kind"] for action in result.canonical_actions]
    assert "clean" in action_kinds
    assert "drop_columns" in action_kinds
