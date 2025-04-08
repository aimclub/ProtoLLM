import json
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from protollm.connectors.utils import (generate_system_prompt_with_schema,
                                       generate_system_prompt_with_tools,
                                       parse_function_calls,
                                       parse_custom_structure,
                                       handle_system_prompt)


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
            "messages": self._convert_messages_to_rest_server_messages(messages),
            **kwargs
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
        
        system_prompt = ""
        if self._tools:
            system_prompt = generate_system_prompt_with_tools(self._tools, self._tool_choice_mode)
        if self._response_format:
            system_prompt = generate_system_prompt_with_schema(self._response_format)
        
        messages = handle_system_prompt(messages, system_prompt)
        
        response = super().invoke(messages, *args, **kwargs)
        
        match response:
            case AIMessage() if ("<function=" in response.content):
                tool_calls = parse_function_calls(response.content)
                if tool_calls:
                    response.tool_calls = tool_calls
                    response.content = ""
            case AIMessage() if self._response_format:
                response = parse_custom_structure(self._response_format, response)
        
        return response
    
    def bind_tools(self, *args, **kwargs: Any) -> Runnable:
        self._tools = kwargs.get("tools", [])
        self._tool_choice_mode = kwargs.get("tool_choice", "auto")
        return self
    
    def with_structured_output(self, *args, **kwargs: Any) -> Runnable:
        self._response_format = kwargs.get("schema", [])
        return self
