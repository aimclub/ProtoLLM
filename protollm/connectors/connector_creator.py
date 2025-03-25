import json
import os
from typing import Any, Dict, List
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from protollm.connectors.utils import (get_access_token,
                                       models_without_function_calling,
                                       models_without_structured_output)
from protollm.definitions import CONFIG_PATH


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
            system_prompt = self._generate_system_prompt_with_tools()
            messages = self._handle_system_prompt(messages, system_prompt)
        
        if self._requires_custom_handling_for_structured_output() and self._response_format:
            system_prompt = self._generate_system_prompt_with_schema()
            messages = self._handle_system_prompt(messages, system_prompt)

        response = self._super_invoke(messages, *args, **kwargs)

        match response:
            case AIMessage() if ("<function=" in response.content):
                tool_calls = self._parse_function_calls(response.content)
                if tool_calls:
                    response.tool_calls = tool_calls
                    response.content = ""
            case AIMessage() if self._response_format:
                response = self._parse_custom_structure(response)

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

    def _generate_system_prompt_with_tools(self) -> str:
        """
        Generates a system prompt with function descriptions and instructions for the model.
        
        Returns:
            System prompt with instructions for calling functions and descriptions of the functions themselves.
        
        Raises:
            ValueError: If tools in an unsupported format have been passed.
        """
        tool_descriptions = []
        match self._tool_choice_mode:
            case "auto" | None | "any" | "required" | True:
                tool_choice_mode = str(self._tool_choice_mode)
            case _:
                tool_choice_mode = f"<<{self._tool_choice_mode}>>"
        for tool in self._tools:
            match tool:
                case dict():
                    tool_descriptions.append(
                        f"Function name: {tool['name']}\n"
                        f"Description: {tool['description']}\n"
                        f"Parameters: {json.dumps(tool['parameters'], ensure_ascii=False)}"
                    )
                case BaseTool():
                    tool_descriptions.append(
                        f"Function name: {tool.name}\n"
                        f"Description: {tool.description}\n"
                        f"Parameters: {json.dumps(tool.args, ensure_ascii=False)}"
                    )
                case _:
                    raise ValueError(
                        "Unsupported tool type. Try using a dictionary or function with the @tool decorator as tools"
                    )
        tool_prefix = "You have access to the following functions:\n\n"
        tool_instructions = (
            "There are the following 4 function call options:\n"
            "- str of the form <<tool_name>>: call <<tool_name>> tool.\n"
            "- 'auto': automatically select a tool (including no tool).\n"
            "- 'none': don't call a tool.\n"
            "- 'any' or 'required' or 'True': at least one tool have to be called.\n\n"
            f"User-selected option - {tool_choice_mode}\n\n"
            "If you choose to call a function ONLY reply in the following format with no prefix or suffix:\n"
            '<function=example_function_name>{"example_name": "example_value"}</function>'
        )
        return tool_prefix + "\n\n".join(tool_descriptions) + "\n\n" + tool_instructions
    
    def _generate_system_prompt_with_schema(self) -> str:
        """
        Generates a system prompt with response format descriptions and instructions for the model.
        
        Returns:
            A system prompt with instructions for structured output and descriptions of the response formats themselves.
            
        Raises:
            ValueError: If the structure descriptions for the response were passed in an unsupported format.
        """
        schema_descriptions = []
        match self._response_format:
            case list():
                schemas = self._response_format
            case _:
                schemas = [self._response_format]
        for schema in schemas:
            match schema:
                case dict():
                    schema_descriptions.append(str(schema))
                case _ if issubclass(schema, BaseModel):
                    schema_descriptions.append(str(schema.model_json_schema()))
                case _:
                    raise ValueError(
                        "Unsupported schema type. Try using a description of the answer structure as a dictionary or"
                        " Pydantic model."
                    )
        schema_prefix = "Generate a JSON object that matches one of the following schemas:\n\n"
        schema_instructions = (
            "Your response must contain ONLY valid JSON, parsable by a standard JSON parser. Do not include any"
            " additional text, explanations, or comments."
        )
        return schema_prefix + "\n\n".join(schema_descriptions) + "\n\n" + schema_instructions

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
    
    def _parse_custom_structure(self, response_from_model) -> dict | BaseModel | None:
        """
        Parses the model response into a dictionary or Pydantic class
        
        Args:
            response_from_model: response of a model that does not support structured output by default
        
        Raises:
            ValueError: If a structured response is not obtained
        """
        match [self._response_format][0]:
            case dict():
                try:
                    return json.loads(response_from_model.content)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        "Failed to return structured output. There may have been a problem with loading JSON from the"
                        f" model.\n{e}"
                    )
            case _ if issubclass([self._response_format][0], BaseModel):
                for schema in [self._response_format]:
                    try:
                        return schema.model_validate_json(response_from_model.content)
                    except ValidationError:
                        continue
                raise ValueError(
                    "Failed to return structured output. There may have been a problem with validating JSON from the"
                    " model."
                )
        
    @staticmethod
    def _parse_function_calls(content: str) -> List[Dict[str, Any]]:
        """
        Parses LLM answer (HTML string) to extract function calls.

        Args:
            content: model response as an HTML string

        Returns:
            A list of dictionaries in tool_calls format
            
        Raises:
            ValueError: If the arguments for a function call are returned in an incorrect format
        """
        tool_calls = []
        pattern = r"<function=(.*?)>(.*?)</function>"
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            function_name, function_args = match
            try:
                arguments = json.loads(function_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error when decoding function arguments: {e}")

            tool_call = {
                "id": f"call_{len(tool_calls) + 1}",
                "type": "tool_call",
                "name": function_name,
                "args": arguments
            }
            tool_calls.append(tool_call)

        return tool_calls
    
    @staticmethod
    def _handle_system_prompt(msgs, sys_prompt):
        match msgs:
            case str():
                return [SystemMessage(content=sys_prompt), HumanMessage(content=msgs)]
            case list():
                if not any(isinstance(msg, SystemMessage) for msg in msgs):
                    msgs.insert(0, SystemMessage(content=sys_prompt))
                else:
                    idx = next((index for index, obj in enumerate(msgs) if isinstance(obj, SystemMessage)), 0)
                    msgs[idx].content += "\n\n" + sys_prompt
        return msgs


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
    if "vsegpt" in model_url:
        base_url, model_name = model_url.split(";")
        api_key = os.getenv("VSE_GPT_KEY")
        return CustomChatOpenAI(model_name=model_name, base_url=base_url, api_key=api_key, *args, **kwargs)
    elif "gigachat" in model_url:
        model_name = model_url.split(";")[1]
        access_token = get_access_token()
        return GigaChat(model=model_name, access_token=access_token, *args, **kwargs)
    elif "api.openai" in model_url:
        model_name = model_url.split(";")[1]
        return ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_KEY"), *args, **kwargs)
    elif "ollama" in model_url:
        url_and_name = model_url.split(";")
        return ChatOllama(model=url_and_name[2], base_url=url_and_name[1], *args, **kwargs)
    elif model_url == "test_model":
        return CustomChatOpenAI(model_name=model_url, api_key="test")
    else:
        raise ValueError("Unsupported provider URL")
    # Possible to add another LangChain compatible connector
