import logging
from typing import Any, Dict, List, Optional

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (AIMessage, BaseMessage, ChatGeneration,
                              ChatResult, HumanMessage, SystemMessage)
from pydantic import PrivateAttr


# Define the custom LLM class
class Llama31ChatModel(BaseChatModel):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.5
    max_tokens: int = 3000
    logging_level: int = logging.INFO

    _logger: logging.Logger = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # Allows arbitrary types like the logger

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=self.logging_level)

    @property
    def _llm_type(self) -> str:
        return "llama31"

    def _prepare_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_context(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        role_map = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system",
        }

        return [
            {"role": role_map.get(type(message), "user"), "content": message.content}
            for message in messages
        ]

    def _prepare_payload(
        self,
        context: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": context,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if stop is not None:
            payload["stop"] = stop
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = self._prepare_headers()
        context = self._prepare_context(messages)
        payload = self._prepare_payload(context, stop, **kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            assistant_response = response.json()["choices"][0]["message"]["content"]
            self._logger.info("REQUEST: %s", payload)
            self._logger.info("RESPONSE: %s", assistant_response)
            ai_message = AIMessage(content=assistant_response)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
        except requests.RequestException as e:
            self._logger.error("API request failed: %s", e)
            raise RuntimeError(f"API request failed: {e}")
