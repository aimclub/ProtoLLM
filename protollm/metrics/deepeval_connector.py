import os

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from openai._types import NOT_GIVEN

from ..connectors import create_llm_connector, get_allowed_providers


class DeepEvalConnector(DeepEvalBaseLLM):
    """Implementation of Evaluation agent based on large language model for Assistant's answers evaluation.

    Uses the LangChain's ChatModel to make requests to a compatible API. Must inherit from the base class and
    implement a set of methods.
    The vsegpt.ru service is used by default, so in the configuration file it is necessary to specify the API key of
    this service and the model name, as in the general case.
    """

    def __init__(self, sys_prompt: str = "", *args, **kwargs):
        """Initialize instance with evaluation LLM.

        Args:
            sys_prompt: predefined rules for model
        """
        super().__init__(*args, **kwargs)
        self._sys_prompt = sys_prompt
        self.model = self.load_model()

    @staticmethod
    def load_model() -> BaseChatModel:
        """Returns LangChain's ChatModel for requests"""
        return create_llm_connector(
            os.getenv("DEEPEVAL_LLM_URL", "test_model"),
            extra_body={"provider": {"only": get_allowed_providers()}}
        )

    def generate(
            self,
            prompt: str,
            *args,
            **kwargs,
    ) -> str | BaseModel:
        """Get a response from LLM to given question.

        Args:
            prompt (str): Query, the model must answer.

        Returns:
            str: Model's response.
        """
        messages = [
            {"role": "system", "content": self._sys_prompt},
            {"role": "user", "content": prompt},
        ]
        response_format = kwargs.get("schema", NOT_GIVEN)
        if response_format == NOT_GIVEN:
            return self.model.invoke(messages).content
        else:
            struct_llm = self.model.with_structured_output(schema=response_format, method="json_mode")
            return struct_llm.invoke(messages)

    async def a_generate(
            self,
            prompt: str,
            *args,
            **kwargs,
    ) -> str:
        """Same as synchronous generate method just because it must be implemented"""
        return self.generate(
            prompt, *args, **kwargs
        )

    def get_model_name(self, *args, **kwargs) -> str:
        """Returns a description of what the class is about"""
        return "Implementation of custom LLM connector using OpenAI compatible API for evaluation."
