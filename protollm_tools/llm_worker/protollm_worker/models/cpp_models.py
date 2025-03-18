import logging

from llama_cpp import Llama
from protollm_sdk.models.job_context_models import (
    PromptModel,
    ChatCompletionModel,
    PromptTransactionModel,
    ChatCompletionTransactionModel,
    PromptTypes,
)

from protollm_worker.models.base import BaseLLM, LocalLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CppModel(LocalLLM, BaseLLM):
    def __init__(self, model_path, n_ctx=8192):
        super().__init__(model_path)

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx * 2,
            verbose=True,
            n_gpu_layers=-1,
        )
        self.handlers = {
            PromptTypes.SINGLE_GENERATION.value: self.generate,
            PromptTypes.CHAT_COMPLETION.value: self.create_completion,
        }

    def __call__(
        self, transaction: PromptTransactionModel | ChatCompletionTransactionModel
    ):
        prompt_type: PromptTypes = transaction.prompt_type
        func = self.handlers[prompt_type]
        return func(transaction.prompt, **transaction.prompt.meta.model_dump())

    def generate(
        self,
        prompt: PromptModel,
        tokens_limit=None,
        temperature=None,
        repeat_penalty=1.1,
        stop_words=None,
        **kwargs,
    ):
        if temperature is None:
            temperature = 0.5
        if stop_words is None:
            stop_words = []
        logger.info(
            f"start generated from single prompt {prompt.content} and temp {temperature}"
        )
        generated_text = self.model(
            prompt.content,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            max_tokens=tokens_limit,
            stop=stop_words,
        )
        response = generated_text["choices"][0]["text"]
        return response

    def create_completion(
        self,
        prompt: ChatCompletionModel,
        tokens_limit=None,
        temperature=None,
        repeat_penalty=1.1,
        stop_words=None,
        **kwargs,
    ):
        if temperature is None:
            temperature = 0.5
        if stop_words is None:
            stop_words = []
        logger.info(f"start generated from chat completion {prompt.messages}")
        messages = prompt.model_dump()["messages"]
        response = self.model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            max_tokens=tokens_limit,
            stop=stop_words,
        )
        return response["choices"][0]["message"]["content"]
