import json
import os
import re
from typing import List, Dict, Any
import uuid

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
import requests


def get_access_token() -> str:
    """
    Gets the access token by the authorisation key specified in the config.
    The token is valid for 30 minutes.

    Returns:
        Access token for Gigachat API
    """
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    request_id = uuid.uuid4()
    authorization_key = os.getenv("AUTHORIZATION_KEY")
    
    payload = {
        'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': f'{request_id}',
        'Authorization': f'Basic {authorization_key}'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)['access_token']


# List of models that do NOT support calling functions out-of-the-box yet
models_without_function_calling = ["r1", "deepseek-chat-alt", "test_model"]

# List of models that do NOT support structured outputs out-of-the-box yet
models_without_structured_output = ["deepseek-chat", "deepseek-chat-alt", "llama-3.3-70b-instruct", "test_model"]


def generate_system_prompt_with_tools(tools: List[Dict | BaseTool], tool_choice_mode: str) -> str:
    """
    Generates a system prompt with function descriptions and instructions for the model.
    
    Args:
        tools: tool list for model
        tool_choice_mode: an indication to the model whether she has to choose a particular tool

    Returns:
        System prompt with instructions for calling functions and descriptions of the functions themselves.

    Raises:
        ValueError: If tools in an unsupported format have been passed.
    """
    tool_descriptions = []
    match tool_choice_mode:
        case "auto" | None | "any" | "required" | True:
            tool_choice_mode = str(tool_choice_mode)
        case _:
            tool_choice_mode = f"<<{tool_choice_mode}>>"
    for tool in tools:
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


def generate_system_prompt_with_schema(response_format) -> str:
    """
    Generates a system prompt with response format descriptions and instructions for the model.
    
    Args:
        response_format: response format as a dictionary or pydantic model

    Returns:
        A system prompt with instructions for structured output and descriptions of the response formats themselves.

    Raises:
        ValueError: If the structure descriptions for the response were passed in an unsupported format.
    """
    schema_descriptions = []
    match response_format:
        case list():
            schemas = response_format
        case _:
            schemas = [response_format]
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


def parse_custom_structure(response_format, response_from_model) -> dict | BaseModel | None:
    """
    Parses the model response into a dictionary or Pydantic class

    Args:
        response_format: response format as a dictionary or Pydantic model
        response_from_model: response of a model that does not support structured output by default

    Raises:
        ValueError: If a structured response is not obtained
    """
    match [response_format][0]:
        case dict():
            try:
                parser = JsonOutputParser()
                return parser.invoke(response_from_model)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Failed to return structured output. There may have been a problem with loading JSON from the"
                    f" model.\n{e}"
                )
        case _ if issubclass([response_format][0], BaseModel):
            for schema in [response_format]:
                try:
                    parser = PydanticOutputParser(pydantic_object=schema)
                    return parser.invoke(response_from_model)
                except ValidationError:
                    continue
            raise ValueError(
                "Failed to return structured output. There may have been a problem with validating JSON from the"
                " model."
            )


def parse_function_calls(content: str) -> List[Dict[str, Any]]:
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
    

def handle_system_prompt(msgs, sys_prompt):
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