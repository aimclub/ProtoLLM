import os
from abc import ABC
from protollm_sdk.models.job_context_models import (
    PromptTransactionModel,
    ChatCompletionTransactionModel,
    PromptModel,
    ChatCompletionModel,
)


class BaseLLM(ABC):
    """
    Base class for language model abstractions. Provides the interface for generating text
    and handling prompt-based and chat-based transactions.
    """

    def __init__(self, model_path_or_url: str):
        """
        Initialize the language model.

        :param model_path_or_url: Path or URL to the model.
        """
        ...

    def __call__(
        self, transaction: PromptTransactionModel | ChatCompletionTransactionModel
    ) -> str:
        """
        Handle a transaction and return the generated text.

        :param transaction: A transaction object containing either prompt or chat data.
        :return: Generated text as a string.
        """
        pass

    def generate(
        self,
        message: PromptModel,
        tokens_limit=None,
        temperature=None,
        stop_words=None,
    ) -> str:
        """
        Generate text based on a single prompt.

        :param message: The input prompt model.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :return: Generated text as a string.
        """
        ...

    def create_completion(
        self,
        messages: ChatCompletionModel,
        tokens_limit=None,
        temperature=None,
        stop_words=None,
    ) -> str:
        """
        Generate a completion based on chat-like input.

        :param messages: A chat completion model containing conversation history.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :return: Generated text as a string.
        """
        ...


class LocalLLM(BaseLLM, ABC):
    """
    Implementation of a local language model. Assumes the model is stored locally.
    """

    def __init__(self, model_path_or_url: str, n_ctx=8192):
        """
        Initialize the local language model.

        :param model_path_or_url: Path to the locally stored model.
        :param n_ctx: Context size for the model (default is 8192).
        :raises ValueError: If the provided path is invalid.
        """
        if not os.path.exists(model_path_or_url):
            raise ValueError("Invalid model_path_or_url. Must be a valid path")

    def generate(
        self,
        message: PromptModel,
        tokens_limit=None,
        temperature=None,
        repeat_penalty=1.1,
        stop_words=None,
    ) -> str:
        """
        Generate text based on a single prompt using the local model.

        :param message: The input prompt model.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param repeat_penalty: Penalty for repeating tokens (default is 1.1).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :return: Generated text as a string.
        """
        ...

    def create_completion(
        self,
        messages: ChatCompletionModel,
        tokens_limit=None,
        temperature=None,
        repeat_penalty=1.1,
        stop_words=None,
    ) -> str:
        """
        Generate a completion based on chat-like input using the local model.

        :param messages: A chat completion model containing conversation history.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param repeat_penalty: Penalty for repeating tokens (default is 1.1).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :return: Generated text as a string.
        """
        ...


class APIlLLM(BaseLLM, ABC):
    """
    Implementation of a language model that communicates with an external API.
    """

    def __init__(
        self,
        model_path_or_url: str,
        token: str,
        default_model: str = None,
        timeout_sec: int = 10 * 60,
    ):
        """
        Initialize the API-based language model.

        :param model_path_or_url: URL to the API endpoint.
        :param token: Authentication token for the API.
        :param default_model: Default model name to use with the API (optional).
        :param timeout_sec: Timeout for API requests in seconds (default is 600 seconds).
        :raises ValueError: If the provided URL is invalid.
        """
        if not (
            model_path_or_url.startswith("http://")
            or model_path_or_url.startswith("https://")
        ):
            raise ValueError("Invalid model_path_or_url. Must be a valid URL")

    def generate(
        self,
        message: PromptModel,
        tokens_limit=None,
        temperature=None,
        stop_words=None,
        model=None,
    ) -> str:
        """
        Generate text based on a single prompt using the API.

        :param message: The input prompt model.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :param model: Specific model to use for the request (optional).
        :return: Generated text as a string.
        """
        ...

    def create_completion(
        self,
        messages: ChatCompletionModel,
        tokens_limit=None,
        temperature=None,
        stop_words=None,
        model=None,
    ) -> str:
        """
        Generate a completion based on chat-like input using the API.

        :param messages: A chat completion model containing conversation history.
        :param tokens_limit: Maximum token limit for the output text (optional).
        :param temperature: Sampling temperature for randomness (optional).
        :param stop_words: A list of stop words to truncate the output text (optional).
        :param model: Specific model to use for the request (optional).
        :return: Generated text as a string.
        """
        ...
