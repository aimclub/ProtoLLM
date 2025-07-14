import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from protollm.connectors.rest_server import ChatRESTServer
from protollm.connectors.utils import (get_access_token,
                                       models_without_function_calling,
                                       models_without_structured_output,
                                       generate_system_prompt_with_schema,
                                       generate_system_prompt_with_tools,
                                       parse_function_calls,
                                       parse_custom_structure,
                                       handle_system_prompt)
from protollm.definitions import CONFIG_PATH, LLM_SERVICES


load_dotenv(CONFIG_PATH)


class CustomChatOpenAI(ChatOpenAI):
    """
    A class that extends ChatOpenAI class from LangChain to support LLama and other models that do not support
    function calls or structured output by default. This is implemented through custom processing of tool calls and JSON
    schemas for a known list of models.
    
    Methods:
        __init__(*args: Any, **kwargs: Any): Initializes the instance with parent configuration and custom handlers
        invoke(messages: str | list, *args, **kwargs) -> AIMessage | dict | BaseModel: Processes input messages with
            custom tool call handling and structured output parsing
        bind_tools(*args, **kwargs: Any) -> Runnable: Enables function calling capability by binding tool definitions
        with_structured_output(*args, **kwargs: Any) -> Runnable: Configures structured output format using JSON schemas
            or Pydantic models
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._response_format = None
        self._tool_choice_mode = None
        self._tools = None

    def invoke(self, messages: str | list, *args, **kwargs) -> AIMessage | dict | BaseModel:
        
        if self._requires_custom_handling_for_tools() and self._tools:
            system_prompt = generate_system_prompt_with_tools(self._tools, self._tool_choice_mode)
            messages = handle_system_prompt(messages, system_prompt)
        
        if self._requires_custom_handling_for_structured_output() and self._response_format:
            system_prompt = generate_system_prompt_with_schema(self._response_format)
            messages = handle_system_prompt(messages, system_prompt)

        response = self._super_invoke(messages, *args, **kwargs)

        match response:
            case AIMessage() if ("<function=" in response.content):
                tool_calls = parse_function_calls(response.content)
                if tool_calls:
                    response.tool_calls = tool_calls
                    response.content = ""
            case AIMessage() if self._response_format:
                response = parse_custom_structure(self._response_format, response)

        return response
    
    def _super_invoke(self, messages, *args, **kwargs):
        return super().invoke(messages, *args, **kwargs)

    def bind_tools(self, *args, **kwargs: Any) -> Runnable:
        if self._requires_custom_handling_for_tools():
            self._tools = kwargs.get("tools", [])
            self._tool_choice_mode = kwargs.get("tool_choice", "auto")
            return self
        else:
            return super().bind_tools(*args, **kwargs)
        
    def with_structured_output(self, *args, **kwargs: Any) -> Runnable:
        if self._requires_custom_handling_for_structured_output():
            self._response_format = kwargs.get("schema", [])
            return self
        else:
            return super().with_structured_output(*args, **kwargs)

    def _requires_custom_handling_for_tools(self) -> bool:
        """
        Determines whether additional processing for tool calling is required for the current model.
        """
        return any(model_name in self.model_name.lower() for model_name in models_without_function_calling)
    
    def _requires_custom_handling_for_structured_output(self) -> bool:
        """
        Determines whether additional processing for structured output is required for the current model.
        """
        return any(model_name in self.model_name.lower() for model_name in models_without_structured_output)


def create_llm_connector(model_url: str, *args: Any, **kwargs: Any) -> CustomChatOpenAI | GigaChat | ChatOpenAI:
    """Creates the proper connector for a given LLM service URL.

    Args:
        model_url: The LLM endpoint for making requests; should be in the format 'base_url;model_endpoint or name'
            - for vsegpt.ru service for example: 'https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct'
            - for Gigachat models family: 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions;Gigachat'
              for Gigachat model you should also install certificates from 'НУЦ Минцифры' -
              instructions - 'https://developers.sber.ru/docs/ru/gigachat/certificates'
            - for OpenAI for example: 'https://api.openai.com/v1;gpt-4o'
            - for Ollama for example: 'ollama;http://localhost:11434;llama3.2'

    Returns:
        The ChatModel object from 'langchain' that can be used to make requests to the LLM service,
        use tools, get structured output.
    """
    if any(service in model_url for service in LLM_SERVICES):
        base_url, model_name = model_url.split(";")
        api_key = os.getenv("LLM_SERVICE_KEY")
        return CustomChatOpenAI(model_name=model_name, base_url=base_url, api_key=api_key, *args, **kwargs)
    elif "gigachat" in model_url:
        model_name = model_url.split(";")[1]
        access_token = get_access_token()
        return GigaChat(model=model_name, access_token=access_token, *args, **kwargs)
    elif "api.openai" in model_url or "groq.com" in model_url:
        base_url, model_name = model_url.split(";")
        return ChatOpenAI(base_url=base_url, model=model_name, api_key=os.getenv("OPENAI_KEY"), *args, **kwargs)
    elif "ollama" in model_url:
        url_and_name = model_url.split(";")
        return ChatOllama(model=url_and_name[2], base_url=url_and_name[1], *args, **kwargs)
    elif "self_hosted" in model_url:
        url_and_name = model_url.split(";")
        return ChatRESTServer(model=url_and_name[2], base_url=url_and_name[1], *args, **kwargs)
    elif model_url == "test_model":
        return CustomChatOpenAI(model_name=model_url, api_key="test")
    else:
        raise ValueError("Unsupported provider URL")
    # Possible to add another LangChain compatible connector
