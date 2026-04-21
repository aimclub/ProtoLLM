from unittest.mock import MagicMock, patch

import pytest

from protollm.agents.builder import GraphBuilder


@pytest.fixture
def conf():
    return {
        "recursion_limit": 10,
        "configurable": {
            "llm": MagicMock(),
            "max_retries": 1,
            "scenario_agents": ["worker_a"],
            "scenario_agent_funcs": {"worker_a": lambda s, c: {**s, "response": "ok"}},
            "tools_for_agents": {"worker_a": ["t"]},
            "tools_descp": "td",
            "prompts": {
                "supervisor": {
                    "problem_statement": "ps",
                    "problem_statement_continue": "psc",
                    "rules": "r",
                    "examples": "ex",
                    "additional_rules": "ar",
                    "enhancemen_significance": "es",
                },
                "planner": {
                    "problem_statement": "ps",
                    "additional_hints": "h",
                    "rules": "r",
                    "examples": "ex",
                    "desc_restrictions": "d",
                },
                "summary": {
                    "problem_statement": "ps",
                    "additional_hints": "h",
                    "rules": "r",
                },
                "chat": {
                    "problem_statement": "ps",
                    "additional_hints": "h",
                },
            },
        },
    }


def test_routing_functions(conf):
    # Arrange
    gb = GraphBuilder(conf)

    # Act & Assert
    assert gb._should_end_chat({"response": "x"}) == "summary"
    assert gb._should_end_chat({}) == "planner"

    assert gb._should_end({"response": "x"}) == "summary"
    assert gb._should_end({"plan": []}) == "summary"
    assert gb._should_end({"plan": ["a"]}) == "supervisor"

    assert gb._routing_function_supervisor({"end": True}) is not None
    assert gb._routing_function_supervisor({}) == "replan_node"

    assert gb._routing_function_planner({"response": "x"}) is not None
    assert gb._routing_function_planner({}) == "supervisor"


def test_stream_runs(conf):
    # Arrange
    gb = GraphBuilder(conf)

    # Patch app.stream to return a simple event sequence
    fake_event = [{"any": {"response": "done"}}]
    gb.app = MagicMock(stream=lambda state, config: [{"k": {"response": "done"}}])

    with patch(
        "protollm.agents.builder.initialize_state",
        return_value={"last_memory": "", "input": "hi"},
    ):
        # Act
        out = list(gb.stream({"input": "hi"}))

    # Assert
    assert len(out) >= 1
    assert out[-1]["response"] == "done"
