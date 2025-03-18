import logging

from protollm_sdk.models.job_context_models import PromptModel, ChatCompletionModel, PromptTransactionModel, \
    ChatCompletionTransactionModel, PromptTypes
from vllm import LLM, SamplingParams

from protollm_worker.models.base import BaseLLM, LocalLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VllMModel(LocalLLM, BaseLLM):
    """
    Implementation of a local language model using vLLM for single prompt generation
    and chat-based completions.
    """

    def __init__(self, model_path, tensor_parallel_size, gpu_memory_utilisation, tokens_len, n_ctx=8192):
        """
        Initialize the vLLM-based model.

        :param model_path: Path to the locally stored model.
        :type model_path: str
        :param n_ctx: Context size for the model (default is 8192).
        :type n_ctx: int
        """
        super().__init__(model_path)

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilisation,
            max_model_len=tokens_len
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

    def generate(
            self,
            prompt: PromptModel,
            tokens_limit=8096,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
            **kwargs
    ) -> str:
        """
        Generate a response for a single prompt.

        :param prompt: Prompt model containing the input text.
        :type prompt: PromptModel
        :param tokens_limit: Maximum token limit for the output (default is 8096).
        :type tokens_limit: int
        :param temperature: Sampling temperature for randomness (default is 0.2).
        :type temperature: float
        :param repeat_penalty: Penalty for repeating tokens (default is 1.1).
        :type repeat_penalty: float
        :param stop_words: List of stop words to truncate the output text (default is an empty list).
        :type stop_words: list or None
        :param kwargs: Additional keyword arguments.
        :return: Generated text as a string.
        :rtype: str
        """
        if temperature is None:
            temperature = 0.2
        if stop_words is None:
            stop_words = []
        if tokens_limit is None or tokens_limit < 1:
            tokens_limit = 8096
        logger.info(f"Starting generation for single prompt: {prompt.content} with temperature {temperature}")

        generated_text = self.model.generate(
            prompt.content, SamplingParams(
                temperature=temperature,
                max_tokens=tokens_limit,
                stop=stop_words,
            )
        )
        return generated_text[0].outputs[0].text

    def create_completion(
            self,
            prompt: ChatCompletionModel,
            tokens_limit=8096,
            temperature=None,
            repeat_penalty=1.1,
            stop_words=None,
            **kwargs
    ) -> str:
        """
        Generate a response for a chat-like prompt with message history.

        :param prompt: Chat completion model containing the conversation history.
        :type prompt: ChatCompletionModel
        :param tokens_limit: Maximum token limit for the output (default is 8096).
        :type tokens_limit: int
        :param temperature: Sampling temperature for randomness (default is 0.2).
        :type temperature: float
        :param repeat_penalty: Penalty for repeating tokens (default is 1.1).
        :type repeat_penalty: float
        :param stop_words: List of stop words to truncate the output text (default is an empty list).
        :type stop_words: list or None
        :param kwargs: Additional keyword arguments.
        :return: Generated text as a string.
        :rtype: str
        """
        if temperature is None:
            temperature = 0.2
        if stop_words is None:
            stop_words = []
        if tokens_limit is None or tokens_limit < 1:
            tokens_limit = 8096
        logger.info(f"Starting generation for chat completion: {prompt.messages}")

        messages = prompt.model_dump()['messages']
        response = self.model.chat(
            messages,
            SamplingParams(
                temperature=temperature,
                max_tokens=tokens_limit,
                stop=stop_words,
            )
        )
        return response[0].outputs[0].text
