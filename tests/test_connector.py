from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pytest

from protollm.connectors import create_llm_connector
from protollm.connectors import CustomChatOpenAI
from protollm.connectors import ChatRESTServer


@pytest.fixture
def custom_chat_openai_without_fc_and_so():
    return CustomChatOpenAI(model_name="test_model", api_key="test")


@pytest.fixture
def custom_chat_openai_with_fc_and_so():
    return CustomChatOpenAI(model_name="test", api_key="test")


@pytest.fixture
def final_messages_for_tool_calling():
    return [
        SystemMessage(content="""You have access to the following functions:

Function name: example_function
Description: Example function
Parameters: {}

There are the following 4 function call options:
- str of the form <<tool_name>>: call <<tool_name>> tool.
- 'auto': automatically select a tool (including no tool).
- 'none': don't call a tool.
- 'any' or 'required' or 'True': at least one tool have to be called.

User-selected option - auto

If you choose to call a function ONLY reply in the following format with no prefix or suffix:
<function=example_function_name>{"example_name": "example_value"}</function>"""),
        HumanMessage(content="Call example_function")
    ]


@pytest.fixture
def dict_schema():
    return {
        "title": "example_schema",
        "description": "Example schema for test",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Person name",
            },
            "age": {
                "type": "int",
                "description": "Person age",
            },
        },
        "required": ["name", "age"],
    }


class ExampleModel(BaseModel):
    """Example"""
    name: str = Field(description="Person name")
    age: int = Field(description="Person age")


def test_connector():
    conn = ChatRESTServer()
    conn.base_url = 'mock'
    chat = conn.create_chat(messages=[HumanMessage('M1'), HumanMessage('M2'), HumanMessage('M3')])
    assert chat is not None


# Basic invoke
def test_invoke_basic(custom_chat_openai_with_fc_and_so):

    mock_response = AIMessage(content="Hello, world!")
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = custom_chat_openai_with_fc_and_so.invoke("Hello")
        assert result.content == "Hello, world!"


def test_invoke_special(custom_chat_openai_without_fc_and_so):

    mock_response = AIMessage(content="Hello, world!")
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = custom_chat_openai_without_fc_and_so.invoke("Hello")
        assert result.content == "Hello, world!"


# Function calling tests for models that doesn't support it out-of-the-box
def test_function_calling_with_dict(custom_chat_openai_without_fc_and_so, final_messages_for_tool_calling):
    tools = [{"name": "example_function", "description": "Example function", "parameters": {}}]
    choice_mode = "auto"
    
    model_with_tools = custom_chat_openai_without_fc_and_so.bind_tools(tools=tools, tool_choice=choice_mode)

    mock_response = AIMessage(content='<function=example_function>{"param": "value"}</function>')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response) as mocked_invoke:
        result = model_with_tools.invoke("Call example_function")
        mocked_invoke.assert_called_with(final_messages_for_tool_calling)
        assert hasattr(result, 'tool_calls')
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]['name'] == "example_function"
        

def test_function_calling_with_tool(custom_chat_openai_without_fc_and_so, final_messages_for_tool_calling):
    @tool
    def example_function():
        """Example function"""
        pass
    
    model_with_tools = custom_chat_openai_without_fc_and_so.bind_tools(tools=[example_function], tool_choice="auto")

    mock_response = AIMessage(content='<function=example_function>{"param": "value"}</function>')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response) as mocked_invoke:
        result = model_with_tools.invoke("Call example_function")
        mocked_invoke.assert_called_with(final_messages_for_tool_calling)
        assert hasattr(result, 'tool_calls')
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]['name'] == "example_function"
        

# Function calling tests for models that support it out-of-the-box
def test_function_calling_with_dict_out_of_the_box(custom_chat_openai_with_fc_and_so):
    tools = [{"name": "example_function", "description": "Example function", "parameters": {}}]
    choice_mode = "auto"
    
    model_with_tools = custom_chat_openai_with_fc_and_so.bind_tools(tools=tools, tool_choice=choice_mode)
    
    mock_response = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    'id': '1',
                    'function': {
                        'arguments': '{"param":"value"}',
                        'name': 'example_function'
                    },
                    'type': 'function'
                }
            ],
        }
    )
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_tools.invoke("Call example_function")
        assert hasattr(result, 'tool_calls')
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]['name'] == "example_function"


def test_function_calling_with_tool_out_of_the_box(custom_chat_openai_with_fc_and_so):
    @tool
    def example_function():
        """Example function"""
        pass
    
    model_with_tools = custom_chat_openai_with_fc_and_so.bind_tools(tools=[example_function], tool_choice="auto")
    
    mock_response = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    'id': '1',
                    'function': {
                        'arguments': '{"param":"value"}',
                        'name': 'example_function'
                    },
                    'type': 'function'
                }
            ],
        }
    )
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_tools.invoke("Call example_function")
        assert hasattr(result, 'tool_calls')
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]['name'] == "example_function"


# Structured output tests for models that doesn't support it out-of-the-box yet
def test_structured_output_pydantic(custom_chat_openai_without_fc_and_so):
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=ExampleModel)
    
    final_messages = [
        SystemMessage(content="Generate a JSON object that matches one of the following schemas:\n\n"
                              "{'description': 'Example', 'properties': {'name': {'description': 'Person name', "
                              "'title': 'Name', 'type': 'string'}, 'age': {'description': 'Person age', "
                              "'title': 'Age', 'type': 'integer'}}, 'required': ['name',"
                              " 'age'], 'title': 'ExampleModel', 'type': 'object'}\n\n"
                              "Your response must contain ONLY valid JSON, parsable by a standard JSON parser. Do not"
                              " include any additional text, explanations, or comments."),
        HumanMessage(content="Generate structured output")
    ]

    mock_response = AIMessage(content='{"name": "John", "age": "30"}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response) as mocked_invoke:
        result = model_with_so.invoke("Generate structured output")
        mocked_invoke.assert_called_with(final_messages)
        assert isinstance(result, ExampleModel)
        assert result.name == "John"
        assert result.age == 30
        

def test_structured_output_dict(custom_chat_openai_without_fc_and_so, dict_schema):
    
    final_messages=[
        SystemMessage(content="Generate a JSON object that matches one of the following schemas:\n\n"
                              "{'title': 'example_schema', 'description': 'Example schema for test', 'type': 'object',"
                              " 'properties': {'name': {'type': 'string', 'description': 'Person name'}, 'age':"
                              " {'type': 'int', 'description': 'Person age'}}, 'required': ['name', 'age']}\n\n"
                              "Your response must contain ONLY valid JSON, parsable by a standard JSON parser. Do not"
                              " include any additional text, explanations, or comments."),
        HumanMessage(content="Generate structured output")
    ]
    
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=dict_schema)
    mock_response = AIMessage(content='{"name": "John", "age": 30}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response) as mocked_invoke:
        result = model_with_so.invoke("Generate structured output")
        mocked_invoke.assert_called_with(final_messages)
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30


def test_structured_output_error_pydantic(custom_chat_openai_without_fc_and_so):
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=ExampleModel)

    mock_response = AIMessage(content='{"name": "John"}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        with pytest.raises(ValueError):
            model_with_so.invoke("Generate structured output")
            

def test_structured_output_error_dict(custom_chat_openai_without_fc_and_so, dict_schema):
    
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=dict_schema)

    mock_response = AIMessage(content='{"name": "John":}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        with pytest.raises(ValueError):
            model_with_so.invoke("Generate structured output")


# Structured output tests for models that support it out-of-the-box
def test_structured_output_pydantic_out_of_the_box(custom_chat_openai_with_fc_and_so):
    model_with_so = custom_chat_openai_with_fc_and_so.with_structured_output(schema=ExampleModel)
    
    mock_response = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        'id': '1',
                        'function': {
                            'arguments': '{"name":"test","age":"30"}',
                            'name': 'ExampleModel'
                        },
                        'type': 'function'
                    }
                ],
                "parsed": ExampleModel.model_validate_json('{"name": "John", "age": 30}')
            },
        )
        
        # 'parsed': ExampleModel.model_validate_json('{"name": "John", "age": 30}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_so.invoke("Generate structured output")
        assert isinstance(result, ExampleModel)
        assert result.name == "John"
        assert result.age == 30


def test_structured_output_dict_out_of_the_box(custom_chat_openai_with_fc_and_so):
    dict_schema = {
        "title": "example_schema",
        "description": "Example schema for test",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Person name",
            },
            "age": {
                "type": "int",
                "description": "Person age",
            },
        },
        "required": ["name", "age"],
    }
    
    model_with_so = custom_chat_openai_with_fc_and_so.with_structured_output(schema=dict_schema)
    mock_response = '{"name": "John", "age": 30}'
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_so.invoke("Generate structured output")
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30


@pytest.mark.parametrize(
    "model_url",
    [
        "https://api.vsegpt.ru/v1;openai/gpt-4o-mini",
        "https://gigachat.devices.sberbank.ru/api/v1/chat/completions;GigaChat",
        "test_model",
        "https://example.com/v1;test/example_model"
    ]
)
def test_connector_creator(model_url):
    with pytest.raises(Exception):
        connector = create_llm_connector(model_url)
        assert issubclass(connector, BaseChatModel)
