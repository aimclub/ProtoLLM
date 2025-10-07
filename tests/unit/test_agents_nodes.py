import types
from typing import Any, Dict, cast
from unittest.mock import MagicMock, patch

import pytest

from protollm.agents.universal_agents import (
    chat_node,
    plan_node,
    replan_node,
    summary_node,
    web_search_node,
)


def _as_state(result: Any) -> Dict[str, Any]:
    """Normalize dict or Command(update=...) to plain state dict for asserts."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "update") and isinstance(getattr(result, "update"), dict):
        return cast(Dict[str, Any], getattr(result, "update"))
    return {}


@pytest.fixture
def base_conf():
    return {
        "configurable": {
            "llm": MagicMock(),
            "max_retries": 1,
            "tools_descp": "tools",
            "prompts": {
                "planner": {
                    "problem_statement": "ps",
                    "additional_hints": "hints",
                    "rules": "rules",
                    "examples": "ex",
                    "desc_restrictions": "dr",
                },
                "replanner": {
                    "problem_statement": "ps",
                    "additional_hints": "hints",
                    "rules": "rules",
                    "examples": "ex",
                },
                "summary": {
                    "problem_statement": "ps",
                    "additional_hints": "hints",
                    "rules": "rules",
                },
                "chat": {
                    "problem_statement": "ps",
                    "additional_hints": "hints",
                },
                "supervisor": {
                    "problem_statement": "ps",
                    "problem_statement_continue": "psc",
                    "rules": "rules",
                    "examples": "ex",
                    "additional_rules": "ar",
                    "enhancemen_significance": "ens",
                },
            },
        }
    }


def test_plan_node_basic(base_conf):
    # Arrange
    state = {"input": "do something", "last_memory": "lm"}
    mock_llm = MagicMock()
    base_conf["configurable"]["llm"] = mock_llm
    mock_llm.invoke.return_value = types.SimpleNamespace(content="{}")

    class Plan:
        def __init__(self, steps):
            self.steps = steps

    with (
        patch("protollm.agents.universal_agents.build_planner_prompt") as build_prompt,
    ):
        build_prompt.return_value = MagicMock(__or__=lambda self, other: self)
        build_prompt.return_value.__or__ = lambda self, other: self
        # Simulate planner.invoke
        build_prompt.return_value.invoke = lambda payload: Plan(["step1", ["a", "b"]])

        # Act
        new_state = cast(
            Dict[str, Any], plan_node(cast(Dict[str, Any], state), base_conf)
        )

    # Assert
    assert new_state["plan"] == ["step1", ["a", "b"]]


def test_plan_node_parser_recovery(base_conf):
    # Arrange
    state = {"input": "do", "last_memory": "lm"}
    base_conf["configurable"]["llm"] = MagicMock()

    with (
        patch("protollm.agents.universal_agents.build_planner_prompt") as build_prompt,
    ):
        build_prompt.return_value = MagicMock()
        build_prompt.return_value.__or__ = lambda self, other: self
        build_prompt.return_value.invoke = lambda payload: types.SimpleNamespace(
            steps=["s1", "s2"]
        )

        # Act
        result = plan_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state.get("plan") == ["s1", "s2"]


def test_replan_node_response_action(base_conf):
    # Arrange
    state = {"input": "q", "plan": ["s1"], "past_steps": []}
    base_conf["configurable"]["llm"] = MagicMock()

    with (
        patch(
            "protollm.agents.universal_agents.build_replanner_prompt"
        ) as build_prompt,
    ):
        build_prompt.return_value = MagicMock()
        build_prompt.return_value.__or__ = lambda self, other: self
        build_prompt.return_value.invoke = lambda payload: types.SimpleNamespace(
            action="response", response="ok"
        )

        # Act
        result = replan_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state.get("response") == "ok"


def test_replan_node_steps_action(base_conf):
    # Arrange
    state = {"input": "q", "plan": ["s1"], "past_steps": [("t", "r")]}
    base_conf["configurable"]["llm"] = MagicMock()

    with (
        patch(
            "protollm.agents.universal_agents.build_replanner_prompt"
        ) as build_prompt,
    ):
        build_prompt.return_value = MagicMock()
        build_prompt.return_value.__or__ = lambda self, other: self
        build_prompt.return_value.invoke = lambda payload: types.SimpleNamespace(
            action="steps", steps=[["x"], ["y"]]
        )

        # Act
        result = replan_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state["plan"] == [["x"], ["y"]]
    assert new_state["next"] == "supervisor"


def test_summary_node_basic(base_conf):
    # Arrange
    state = {"response": "r", "input": "q", "past_steps": []}
    mock_llm = MagicMock()
    base_conf["configurable"]["llm"] = mock_llm
    mock_llm.invoke.return_value = types.SimpleNamespace(content="summary")

    with patch("protollm.agents.universal_agents.build_summary_prompt") as build_prompt:
        build_prompt.return_value = MagicMock()
        build_prompt.return_value.__or__ = lambda self, other: self
        build_prompt.return_value.invoke = lambda payload: types.SimpleNamespace(
            content="summary"
        )

        # Act
        result = summary_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state["response"] == "summary"


def test_chat_node_text_flow(base_conf):
    # Arrange
    state = {"input": "hi", "last_memory": "", "attached_img": ""}
    mock_llm = MagicMock()
    base_conf["configurable"]["llm"] = mock_llm

    with (
        patch(
            "protollm.agents.universal_agents.chat_parser",
            new=types.SimpleNamespace(
                parse=lambda *_args, **_kwargs: types.SimpleNamespace(
                    action=types.SimpleNamespace(next="planner")
                )
            ),
        ),
        patch("protollm.agents.universal_agents.build_chat_prompt", return_value="sys"),
        patch("protollm.agents.universal_agents.prompt_func", return_value={}),
    ):
        # Act
        result = chat_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state["next"] == "planner"


def test_chat_node_image_flow(base_conf):
    # Arrange
    state = {"input": "hi", "last_memory": "", "attached_img": "/tmp/a.png"}
    base_conf["configurable"]["visual_model"] = MagicMock()

    with (
        patch("protollm.agents.universal_agents.convert_to_base64", return_value="img"),
        patch(
            "protollm.agents.universal_agents.chat_parser",
            new=types.SimpleNamespace(
                parse=lambda *_args, **_kwargs: types.SimpleNamespace(
                    action=types.SimpleNamespace(next="planner")
                )
            ),
        ),
        patch("protollm.agents.universal_agents.build_chat_prompt", return_value="sys"),
        patch("protollm.agents.universal_agents.prompt_func", return_value={}),
    ):
        # Act
        result = chat_node(cast(Dict[str, Any], state), base_conf)
        new_state = _as_state(result)

    # Assert
    assert new_state["next"] == "planner"


def test_web_search_node_success(base_conf):
    # Arrange
    state = {"task": "search something"}
    base_conf["configurable"]["llm"] = MagicMock()
    base_conf["configurable"]["max_retries"] = 1
    base_conf["configurable"]["web_tools"] = []

    with patch("protollm.agents.universal_agents.create_react_agent") as cra:
        fake_agent_resp = {
            "messages": [
                types.SimpleNamespace(type="ai", content=""),
                types.SimpleNamespace(type="ai", content="done"),
            ]
        }
        cra.return_value = MagicMock(invoke=lambda payload: fake_agent_resp)
        # Act
        result = web_search_node(cast(Dict[str, Any], state), base_conf)
        state_after = _as_state(result)

    # Assert
    assert "past_steps" in state_after


def test_web_search_node_retry_on_exception(base_conf):
    # Arrange
    state = {"task": "search"}
    base_conf["configurable"]["llm"] = MagicMock()
    base_conf["configurable"]["max_retries"] = 1

    with patch("protollm.agents.universal_agents.create_react_agent") as cra:
        cra.return_value = MagicMock(invoke=MagicMock(side_effect=Exception("boom")))

        # Act
        result = web_search_node(cast(Dict[str, Any], state), base_conf)
        state_after = _as_state(result)

    # Assert
    assert isinstance(state_after, dict)
