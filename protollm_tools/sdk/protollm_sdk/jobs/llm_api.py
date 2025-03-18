import logging
from urllib.parse import urljoin

import httpx

from protollm_sdk.models.job_context_models import (
    PromptModel,
    ResponseModel,
    ChatCompletionModel,
)

logger = logging.getLogger(__name__)


class LLMAPI:
    """
    Class for using large language models deployed on the cluster
    """

    def __init__(
        self,
        llm_api_host: str,
        llm_api_port: str | int | None = None,
        timeout_sec: int = 10 * 60,
    ):
        """
        Initialize LLMAPI

        :param llm_api_host: Host of the LLM API
        :type llm_api_host: str
        :param llm_api_port: Port of the LLM API
        :type llm_api_port: str | int | None
        :param timeout_sec: Timeout in seconds
        :type timeout_sec: int
        """
        self.path = (
            f"http://{llm_api_host}:{llm_api_port}"
            if llm_api_port is not None
            else f"http://{llm_api_host}"
        )
        self.timeout_sec = timeout_sec
        self.client = httpx.Client()

    def inference(self, request: PromptModel) -> ResponseModel:
        """
        Simple response to a request in the response-question format

        :param request: simple request
        :type request: PromptModel
        :return: ResponseModel
        """
        try:
            response = self.client.post(
                urljoin(self.path, "/generate"),
                headers={"Content-type": "application/json"},
                data=request.model_dump_json(),
                timeout=self.timeout_sec,
            )
            if response.status_code == 500:
                raise ConnectionError("The LLM server is not available.")
            elif response.status_code == 422:
                raise ValueError(f"Data model validation error. {response.json()}")
            result = ResponseModel.model_validate(response.json())
            logger.info("The request has been successfully processed.")
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.info(msg)
            raise Exception(msg)

    def chat_completion(self, request: ChatCompletionModel) -> ResponseModel:
        """
        Response to a request taking into account the chat history
        (the initial prompt, user requests and model responses will be sent)

        :param request: request for chat completion
        :type request: ChatCompletionModel
        :return: ResponseModel
        """
        try:
            response = self.client.post(
                urljoin(self.path, "/chat_completion"),
                headers={"Content-type": "application/json"},
                data=request.model_dump_json(),
                timeout=self.timeout_sec,
            )
            if response.status_code == 500:
                raise ConnectionError("The LLM server is not available.")
            elif response.status_code == 422:
                raise ValueError(f"Data model validation error. {response.json()}")
            result = ResponseModel.model_validate(response.json())
            logger.info("The request has been successfully processed.")
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.info(msg)
            raise Exception(msg)

    def __del__(self):
        """
        Close client

        :return: None
        """
        self.client.close()
