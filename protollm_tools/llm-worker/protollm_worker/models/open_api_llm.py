import logging
from time import sleep

from protollm_worker.models.base import APIlLLM, BaseLLM
from protollm_sdk.models.job_context_models import PromptTypes, PromptTransactionModel, ChatCompletionTransactionModel, \
    PromptModel, ChatCompletionModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAPILLM(APIlLLM, BaseLLM):
    """
    Implementation of a language model that interacts with the OpenAI API.
    Handles both single-prompt generations and chat completions.
    """

    def __init__(self, model_url: str, token: str, default_model: str = None, timeout_sec: int = 10 * 60):
        """
        Initialize the OpenAPILLM with API credentials and default parameters.

        :param model_url: URL of the OpenAI API endpoint.
        :type model_url: str
        :param token: API key for authentication.
        :type token: str
        :param default_model: Default model name to use for completions (optional).
        :type default_model: str
        :param timeout_sec: Timeout for API requests in seconds (default is 10 minutes).
        :type timeout_sec: int
        """
        super().__init__(model_url, token)

        self.model = default_model
        self.token = token
        self.timeout_sec = timeout_sec
        self.client = OpenAI(
            api_key=token,
            base_url=model_url
        )
        self.handlers = {
            PromptTypes.SINGLE_GENERATION.value: self.generate,
            PromptTypes.CHAT_COMPLETION.value: self.create_completion,
        }

    def __call__(self, transaction: PromptTransactionModel | ChatCompletionTransactionModel):
        """
        Handle a transaction and return the generated text based on the prompt type.

        :param transaction: Transaction object containing the prompt and metadata.
        :type transaction: PromptTransactionModel | ChatCompletionTransactionModel
        :return: Generated text as a string.
        :rtype: str
        """
        prompt_type: PromptTypes = transaction.prompt_type
        func = self.handlers[prompt_type]
        return func(transaction.prompt, **transaction.prompt.meta.model_dump())

    def _chat_completion(
            self,
            messages: list[dict[str, str]],
            temperature: float,
            tokens_limit: int,
            model: str or None
    ) -> str:
        """
        Set up a chat completion request to the OpenAI API.

        :param messages: List of message dictionaries containing roles and content.
        :type messages: list[dict[str, str]]
        :param temperature: Sampling temperature for randomness.
        :type temperature: float
        :param tokens_limit: Maximum number of tokens in the response.
        :type tokens_limit: int
        :param model: Model name to use for the request (optional).
        :type model: str or None
        :return: Generated response text.
        :rtype: str
        """
        sleep(1)  # Throttling to adhere to GPT rate limits
        model = model or self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=tokens_limit,
            timeout=self.timeout_sec)

        result = response.choices[0].message.content
        logger.info(result)
        return result

    def generate(
            self,
            prompt: PromptModel,
            tokens_limit=None,
            temperature=None,
            model=None,
            **kwargs
    ) -> str:
        """
        Generate a response for a single prompt.

        :param prompt: Prompt model containing the input text.
        :type prompt: PromptModel
        :param tokens_limit: Maximum token limit for the output (optional).
        :type tokens_limit: int
        :param temperature: Sampling temperature for randomness (default is 0.5).
        :type temperature: float
        :param model: Specific model to use for the request (optional).
        :type model: str or None
        :return: Generated text as a string.
        :rtype: str
        """
        if temperature is None:
            temperature = 0.5
        messages = [{"role": "user", "content": prompt.content}]
        logger.info(f"Starting generation for single prompt: {prompt.content} with temperature {temperature}")
        try:
            result = self._chat_completion(
                messages,
                temperature,
                tokens_limit,
                model,
            )
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)

    def create_completion(
            self,
            prompt: ChatCompletionModel,
            tokens_limit=None,
            temperature=None,
            model=None,
            **kwargs
    ) -> str:
        """
        Generate a response for a chat-like prompt with message history.

        :param prompt: Chat completion model containing the conversation history.
        :type prompt: ChatCompletionModel
        :param tokens_limit: Maximum token limit for the output (optional).
        :type tokens_limit: int
        :param temperature: Sampling temperature for randomness (default is 0.5).
        :type temperature: float
        :param model: Specific model to use for the request (optional).
        :type model: str or None
        :return: Generated text as a string.
        :rtype: str
        """
        if temperature is None:
            temperature = 0.5
        logger.info(f"Starting generation for chat completion: {prompt.messages}")
        messages = [{"role": unit.role, "content": unit.content} for unit in prompt.messages]
        try:
            result = self._chat_completion(
                messages,
                temperature,
                tokens_limit,
                model
            )
            return result
        except Exception as ex:
            msg = f"The response generation has been interrupted. Error: {ex}."
            logger.error(msg)
            raise Exception(msg)
