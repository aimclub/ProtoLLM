import json
import re
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError


class ChatRESTServer(BaseChatModel):
    model_name: Optional[str] = 'llama'
    base_url: str = 'http://localhost'
    max_tokens: int = 2048
    temperature: float = 0.1
    
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._response_format = None
        self._tool_choice_mode = None
        self._tools = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-rest-server"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
            "url": self.base_url
        }

    @staticmethod
    def _convert_messages_to_rest_server_messages(
            messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        chat_messages: List = []
        for message in messages:
            match message:
                case HumanMessage():
                    role = "user"
                case AIMessage():
                    role = "assistant"
                case SystemMessage():
                    role = "system"
                case _:
                    raise ValueError("Received unsupported message type.")

            if isinstance(message.content, str):
                content = message.content
            else:
                raise ValueError(
                    "Unsupported message content type. "
                    "Must have type 'text' "
                )
            chat_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )
        return chat_messages

    def _create_chat(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        additional_arguments = kwargs
        payload = {
            "job_id": str(uuid4()),
            "priority": None,
            "source": "local",
            "meta": {
                "temperature": self.temperature,
                "tokens_limit": self.max_tokens,
                "stop_words": stop,
                "model": None
            },
            "messages": self._convert_messages_to_rest_server_messages(messages)
        }

        response = requests.post(
            url=f'{self.base_url}/chat_completion',
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.encoding = "utf-8"
        match response.status_code:
            case 200:
                pass
            case 404:
                raise ValueError(
                    "CustomWeb call failed with status code 404. "
                    "Maybe you need to connect to the corporate network."
                )
            case _:
                optional_detail = response.text
                raise ValueError(
                    f"CustomWeb call failed with status code "
                    f"{response.status_code}. "
                    f"Details: {optional_detail}"
                )
        return json.loads(response.text)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        response = self._create_chat(messages, stop, **kwargs)
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=response['content']),
        )
        return ChatResult(generations=[chat_generation])
    
    def invoke(self, messages: str | list, *args, **kwargs) -> AIMessage | dict | BaseModel:
        
        if self._tools:
            system_prompt = self._generate_system_prompt_with_tools()
            messages = self._handle_system_prompt(messages, system_prompt)
        
        if self._response_format:
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
        self._tools = kwargs.get("tools", [])
        self._tool_choice_mode = kwargs.get("tool_choice", "auto")
        return self
    
    def with_structured_output(self, *args, **kwargs: Any) -> Runnable:
        self._response_format = kwargs.get("schema", [])
        return self
    
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
                    parser = JsonOutputParser()
                    return parser.invoke(response_from_model)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        "Failed to return structured output. There may have been a problem with loading JSON from the"
                        f" model.\n{e}"
                    )
            case _ if issubclass([self._response_format][0], BaseModel):
                for schema in [self._response_format]:
                    try:
                        parser = PydanticOutputParser(pydantic_object=schema)
                        return parser.invoke(response_from_model)
                    except ValidationError:
                        continue
                raise ValueError(
                    "Failed to return structured output. There may have been a problem with validating JSON from the"
                    " model."
                )

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
