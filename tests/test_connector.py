from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pytest

from protollm.connectors.connector_creator import create_llm_connector
from protollm.connectors.connector_creator import CustomChatOpenAI
from protollm.connectors.rest_server import ChatRESTServer


@pytest.fixture
def custom_chat_openai_without_fc_and_so():
    return CustomChatOpenAI(model_name="test_model", api_key="test")


@pytest.fixture
def custom_chat_openai_with_fc_and_so():
    return CustomChatOpenAI(model_name="test", api_key="test")


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
def test_function_calling_with_dict(custom_chat_openai_without_fc_and_so):
    tools = [{"name": "example_function", "description": "Example function", "parameters": {}}]
    choice_mode = "auto"
    
    model_with_tools = custom_chat_openai_without_fc_and_so.bind_tools(tools=tools, tool_choice=choice_mode)

    mock_response = AIMessage(content='<function=example_function>{"param": "value"}</function>')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_tools.invoke("Call example_function")
        assert hasattr(result, 'tool_calls')
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]['name'] == "example_function"
        

def test_function_calling_with_tool(custom_chat_openai_without_fc_and_so):
    @tool
    def example_function():
        """Example function"""
        pass
    
    model_with_tools = custom_chat_openai_without_fc_and_so.bind_tools(tools=[example_function], tool_choice="auto")

    mock_response = AIMessage(content='<function=example_function>{"param": "value"}</function>')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_tools.invoke("Call example_function")
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

    mock_response = AIMessage(content='{"name": "John", "age": "30"}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_so.invoke("Generate structured output")
        assert isinstance(result, ExampleModel)
        assert result.name == "John"
        assert result.age == 30
        

def test_structured_output_dict(custom_chat_openai_without_fc_and_so):
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
    
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=dict_schema)
    mock_response = AIMessage(content='{"name": "John", "age": 30}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        result = model_with_so.invoke("Generate structured output")
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30


def test_structured_output_error(custom_chat_openai_without_fc_and_so):
    model_with_so = custom_chat_openai_without_fc_and_so.with_structured_output(schema=ExampleModel)

    mock_response = AIMessage(content='{"name": "John"}')
    with patch.object(CustomChatOpenAI, '_super_invoke', return_value=mock_response):
        with pytest.raises(Exception):
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
