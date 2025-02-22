import logging
import random
from typing import Sequence

from protollm_worker.models.base import BaseLLM
from protollm_sdk.models.job_context_models import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomLLM(BaseLLM):
    """
    Implementation of the stub-model that simulates a llm inference.
    """
    def __init__(
            self,
            mock_generation_pool: Sequence[str] = None,
            name: str = None,
            description: str = None,
            ip_address: str = None
    ) -> None:
        """
        Initializes the stub-model class

        :param mock_generation_pool: stub for answering simulation that contains prepared answers.
        """
        if mock_generation_pool is None:
            self.mock_generation_pool = [
                "Echpochmak is a national dish of Tatar and Bashkir cuisines!",
                "Your plan for the day will be generated tomorrow.",
                "I'm tired of answering you!",
                "It's just another templated answer...",
                "Oracle JDK. The official Java development kit from Oracle. Provides high performance!",
            ]
        else:
            self.mock_generation_pool = mock_generation_pool

        super().__init__(name, description, ip_address)

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

    def generate(self, message: PromptModel, **kwargs) -> str:
        """
        Simulates the answer on a single prompt
        """
        logger.info(f"handled message: {message.content}")
        return random.choice(self.mock_generation_pool)

    def create_completion(self, messages: ChatCompletionModel, **kwargs) -> str:
        """
        Simulates the answer on a single prompt
        """
        logger.info(f"handled messages: {[m.content for m in messages.messages]}")
        return random.choice(self.mock_generation_pool)
