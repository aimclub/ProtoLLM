import logging
from time import sleep

from openai import OpenAI

from protollm_sdk.models.job_context_models import (
    PromptModel,
    ChatCompletionModel,
    ResponseModel,
)

logger = logging.getLogger(__name__)


class OuterLLMAPI:
    """
    Class for using large language models deployed on the vsegpt.ru
    """

    def __init__(
        self,
        key: str,
        timeout_sec: int = 10 * 60,
    ):
        """
        Initialize OuterLLMAPI

        :param timeout_sec: Timeout in seconds
        :type timeout_sec: int
        """
        self.model = "openai/gpt-4o-2024-08-06"
        self.key = key
        self.timeout_sec = timeout_sec
        self.client = OpenAI(
            api_key=self.key,
            base_url="https://api.vsegpt.ru/v1",
        )

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        tokens_limit: int,
        model: str or None,
    ) -> str:
        """
        Set up a chat completion

        :param messages: history of messages
        :type messages: list[dict[str, str]]
        :param temperature: degree of creativity in the responses
        :type temperature: float
        :param tokens_limit: maximum number of tokens in the response
        :type tokens_limit: int
        :param model: model name
        :type model: str or None
        :return: str
        """
        sleep(1)  # TODO remake in singleton
        # GPT chats don't respond more than 1 message at second
        model = model or self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=tokens_limit,
            timeout=self.timeout_sec,
        )

        result = response.choices[0].message.content
        logger.info(result)
        return result

    def inference(self, request: PromptModel) -> ResponseModel:
        """
        Simple response to a request in the response-question format

        :param request: simple request
        :type request: PromptModel
        :return: ResponseModel
        """
        messages = [{"role": "user", "content": request.content}]
        try:
            result = self._chat_completion(
                messages,
                request.meta.temperature,
                request.meta.tokens_limit,
                model=request.meta.model,
            )
            return ResponseModel(content=result)
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)

    def chat_completion(self, request: ChatCompletionModel) -> ResponseModel:
        """
        Response to a request taking into account the chat history
        (the initial prompt, user requests and model responses will be sent)

        :param request: request for chat completion
        :type request: ChatCompletionModel
        :return: ResponseModel
        """
        messages = [
            {"role": unit.role, "content": unit.content} for unit in request.messages
        ]
        try:
            result = self._chat_completion(
                messages,
                request.meta.temperature,
                request.meta.tokens_limit,
                model=request.meta.model,
            )
            return ResponseModel(content=result)
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)
