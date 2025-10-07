from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from protollm.connectors.utils import (
    generate_system_prompt_with_schema,
    generate_system_prompt_with_tools,
    handle_system_prompt,
    parse_custom_structure,
    parse_function_calls,
)


class Example(BaseModel):
    name: str = Field(description="n")
    age: int = Field(description="a")


def test_generate_system_prompt_with_tools_dict():
    # Arrange
    tools: list[dict[str, Any]] = [{"name": "f", "description": "d", "parameters": {}}]

    # Act
    prompt = generate_system_prompt_with_tools(cast(list, tools), "auto")

    # Assert
    assert "Function name: f" in prompt
    assert "User-selected option - auto" in prompt


def test_generate_system_prompt_with_tools_tool():
    # Arrange
    @tool
    def f():
        """d"""
        pass

    # Act
    prompt = generate_system_prompt_with_tools([f], "required")

    # Assert
    assert "Function name: f" in prompt
    assert "required" in prompt


def test_generate_system_prompt_with_schema_pydantic():
    # Act
    prompt = generate_system_prompt_with_schema(Example)
    # Assert
    assert "Generate a JSON object" in prompt
    assert "Example" in prompt


def test_generate_system_prompt_with_schema_dict():
    # Arrange
    schema = {"title": "ex", "type": "object", "properties": {}}
    # Act
    prompt = generate_system_prompt_with_schema(schema)
    # Assert
    assert "ex" in prompt


def test_parse_function_calls_ok():
    # Arrange
    content = '<function=do_it>{"x": 1}</function>'
    # Act
    calls = parse_function_calls(content)
    # Assert
    assert calls[0]["name"] == "do_it"
    assert calls[0]["args"]["x"] == 1


def test_parse_function_calls_bad_json():
    # Arrange
    content = "<function=do_it>{bad}</function>"
    # Act / Assert
    with pytest.raises(ValueError):
        parse_function_calls(content)


def test_parse_custom_structure_pydantic_ok():
    # Arrange
    response = AIMessage(content='{"name": "john", "age": 30}')

    # Act
    obj = parse_custom_structure(Example, response)
    # Assert
    assert isinstance(obj, Example)
    assert obj.age == 30


def test_parse_custom_structure_dict_ok():
    # Arrange
    response = AIMessage(content='{"name": "john", "age": 30}')

    schema = {
        "title": "ex",
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    # Act
    obj = parse_custom_structure(schema, response)
    # Assert
    assert isinstance(obj, dict)


def test_parse_custom_structure_error():
    # Arrange
    class FakeResp:
        content = '{"name": 10}'

    # Assert
    with pytest.raises(ValueError):
        parse_custom_structure(Example, FakeResp())


def test_handle_system_prompt_injection():
    # Arrange
    sys = "SYS"
    # Act
    msgs = handle_system_prompt("hello", sys)
    # Assert
    assert len(msgs) == 2
