import os
import uuid
from dataclasses import dataclass
from typing import Dict

from dataclasses_json import dataclass_json
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from dotenv import load_dotenv
from openai import OpenAI
from openai._types import NOT_GIVEN
from protollm_sdk.jobs.llm_api import LLMAPI
from protollm_sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm_sdk.models.job_context_models import PromptModel


@dataclass_json
@dataclass
class Message:
    """Template for message in following format {"role", "content"}."""

    role: str
    content: str

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


class VseGPTConnector(DeepEvalBaseLLM):
    """Implementation of Evaluation agent based on large language model for Assistant's answers evaluation."""

    def __init__(
            self,
            model: str,
            sys_prompt: str = "",
            base_url="https://api.vsegpt.ru/v1",
    ):
        """Initialize instance with evaluation LLM.

        Args:
            model: Evaluation model's name
            sys_prompt: predefined rules for model
            base_url: URL where models are available
        """
        load_dotenv("./config.env")
        self._sys_prompt = sys_prompt
        self._model_name = model
        self.base_url = base_url
        self.model = self.load_model()

    def load_model(self) -> OpenAI:
        """Load model's instance."""
        # TODO extend pull of possible LLMs (Not only just OpenAI's models)
        return OpenAI(api_key=os.environ.get("VSE_GPT_KEY"), base_url=self.base_url)

    def generate(
            self,
            prompt: str,
            context: str | None = None,
            temperature: float = 0.015,
            chat_history: list[Message] | None = None,
            *args,
            **kwargs,
    ) -> str:
        """Get a response form LLM to given question.

        Args:
            prompt (str): User's question, the model must answer.
            context (str, optional): Supplementary information, may be used for answer.
            temperature (float, optional): Determines randomness and diversity of generated answers.
            The higher the temperature, the more diverse the answer is. Defaults to .015.

        Returns:
            str: Model's response for user's question.
        """
        usr_msg_template = (
            prompt if context is None else f"Вопрос:{prompt} Контекст:{context}"
        )
        if chat_history is None:
            chat_history = []

        messages = [
            {"role": "system", "content": self._sys_prompt},
            *(msg.to_dict() for msg in chat_history),
            {"role": "user", "content": usr_msg_template},
        ]
        response_format = kwargs.get("schema", NOT_GIVEN)
        response = self.model.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=8182,
            response_format=response_format,
        )
        return response.choices[0].message.content

    async def a_generate(
            self,
            prompt: str,
            context: str | None = None,
            temperature: float = 0.015,
            chat_history: list[Message] | None = None,
            *args,
            **kwargs,
    ) -> str:
        return self.generate(
            prompt, context, temperature, chat_history, *args, **kwargs
        )

    def get_model_name(self, *args, **kwargs) -> str:
        return "Implementation of custom LLM for evaluation."


model = VseGPTConnector(model="openai/gpt-4o-mini")
metrics_init_params = {
    "model": model,
    "verbose_mode": False,
    "async_mode": False,
}
correctness_metric = GEval(
    name="Correctness",
    criteria=(
        "1. Correctness and Relevance:"
        "- Compare the actual response against the expected response. Determine the"
        " extent to which the actual response captures the key elements and concepts of"
        " the expected response."
        "- Assign higher scores to actual responses that accurately reflect the core"
        " information of the expected response, even if only partial."
        "2. Numerical Accuracy and Interpretation:"
        "- Pay particular attention to any numerical values present in the expected"
        " response. Verify that these values are correctly included in the actual"
        " response and accurately interpreted within the context."
        "- Ensure that units of measurement, scales, and numerical relationships are"
        " preserved and correctly conveyed."
        "3. Allowance for Partial Information:"
        "- Do not heavily penalize the actual response for incompleteness if it covers"
        " significant aspects of the expected response. Prioritize the correctness of"
        " provided information over total completeness."
        "4. Handling of Extraneous Information:"
        "- While additional information not present in the expected response should not"
        " necessarily reduce score, ensure that such additions do not introduce"
        " inaccuracies or deviate from the context of the expected response."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    **metrics_init_params,
)


def local_llm(question: str, meta: dict, host: str, port: str | int):
    llmapi = LLMAPI(llm_api_host=host, llm_api_port=port)
    llm_request = PromptModel(job_id=str(uuid.uuid4()), meta=meta, content=question)
    res = llmapi.inference(llm_request)
    return res.content


def outer_llm(question: str, meta: dict, key: str):
    llmapi = OuterLLMAPI(key)
    llm_request = PromptModel(job_id=str(uuid.uuid4()), meta=meta, content=question)
    res = llmapi.inference(llm_request)
    return res.content


if __name__ == "__main__":
    load_dotenv("config.env")
    q = "Что необходимо обеспечить в рамках формирования улично-дорожной сети (УДС) Санкт-Петербурга?"
    context = "В рамках указанной задачи необходимо обеспечить формирование опорного каркаса улично-дорожной сети (далее - УДС) Санкт-Петербурга за счет развития современной транспортной инфраструктуры, повышения связности объектов транспортной инфраструктуры, увеличения пропускной способности, развития магистралей непрерывного движения и магистралей с улучшенными условиями движения, вылетных магистралей, строительства многоуровневых развязок и транспортно-пересадочных узлов."
    question = f"Question: {q}; Context: {context}"

    host = "URL"
    port = 6672
    meta = {"temperature": 0.05, "tokens_limit": 4096, "stop_words": None}
    key = os.environ.get("KEY")
    local_llm_ans = local_llm(question, meta, host, port)
    outer_llm_ans = outer_llm(question, meta, key)

    metric_local = correctness_metric.measure(
        LLMTestCase(
            input=question,
            actual_output=local_llm_ans,
            expected_output="В рамках формирования улично-дорожной сети (УДС) Санкт-Петербурга необходимо обеспечить формирование опорного каркаса УДС за счет развития современной транспортной инфраструктуры, повышения связности объектов транспортной инфраструктуры, увеличения пропускной способности, развития магистралей непрерывного движения и магистралей с улучшенными условиями движения, строительства многоуровневых развязок и транспортно-пересадочных узлов.",
            retrieval_context=None,
        )
    )
    metric_outer = correctness_metric.measure(
        LLMTestCase(
            input=question,
            actual_output=outer_llm_ans,
            expected_output="В рамках формирования улично-дорожной сети (УДС) Санкт-Петербурга "
                            "необходимо обеспечить формирование опорного каркаса УДС за счет развития "
                            "современной транспортной инфраструктуры, повышения связности объектов "
                            "транспортной инфраструктуры, увеличения пропускной способности, развития "
                            "магистралей непрерывного движения и магистралей с улучшенными условиями движения, "
                            "строительства многоуровневых развязок и транспортно-пересадочных узлов.",
            retrieval_context=None,
        )
    )

    print(f"Question: {q}")
    if metric_local < metric_outer:
        print(f"Ответ VseGPT LLM: \n {outer_llm_ans}")
        print(f"Metric outer: {metric_outer}")
    else:
        print(f"Ответ локальной LLM: \n {local_llm_ans}")
        print(f"Metric local: {metric_local}")
