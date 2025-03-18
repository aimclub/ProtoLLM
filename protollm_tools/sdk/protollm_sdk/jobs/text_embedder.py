import logging
from urllib.parse import urljoin

import httpx

from protollm_sdk.models.job_context_models import (
    TextEmbedderRequest,
    TextEmbedderResponse,
    ToEmbed,
)

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Class provides an object interface to a text embedder.
    """

    def __init__(
        self,
        text_emb_host: str,
        text_emb_port: str | int | None = None,
        timeout_sec: int = 10 * 60,
    ):
        """
        Initialize TextEmbedder

        :param text_emb_host: host of the text embedder
        :type text_emb_host: str
        :param text_emb_port: port of the text embedder
        :type text_emb_port: str | int | None
        :param timeout_sec: timeout in seconds
        :type timeout_sec: int
        """
        self.path = (
            f"http://{text_emb_host}:{text_emb_port}"
            if text_emb_port is not None
            else f"http://{text_emb_host}"
        )
        self.timeout_sec = timeout_sec
        self.client = httpx.Client()

    def inference(self, request: TextEmbedderRequest) -> TextEmbedderResponse:
        """
        Create an embedding for the input text

        :param request: request
        :type request: TextEmbedderRequest
        :return: TextEmbedderResponse
        """
        try:
            emb = ToEmbed(inputs=request.inputs, truncate=request.truncate)
            response = self.client.post(
                urljoin(self.path, "/embed"),
                headers={"Content-type": "application/json"},
                json=emb.model_dump(),
                timeout=self.timeout_sec,
            )
            if response.status_code == 500:
                raise ConnectionError("The LLM server is not available.")
            elif response.status_code == 422:
                raise ValueError(f"Data model validation error. {response.json()}")
            result = response.json()[0]
            logger.info("The embedding has been completed successfully.")
            text_embedder_result = {"embeddings": result}
            return TextEmbedderResponse(**text_embedder_result)
        except Exception as ex:
            msg = f"The embedding was interrupted. Error: {ex}."
            logger.info(msg)
            raise Exception(msg)
