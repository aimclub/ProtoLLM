from typing import Optional
from unittest.mock import patch

from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from protollm.metrics.deepeval_connector import DeepEvalConnector
from protollm.metrics.evaluation_metrics import correctness_metric


class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


def test_metric_connector():
    model = DeepEvalConnector()
    mock_response = AIMessage(content="Hello, world!")
    with patch.object(model, 'generate', return_value=mock_response):
        result = model.generate("Hello")
        assert result.content == "Hello, world!"


def test_metric_connector_with_schema():
    model = DeepEvalConnector()
    mock_response = Joke.model_validate_json('{"setup": "test", "punchline": "test", "score": "7"}')
    with patch.object(model, 'generate', return_value=mock_response):
        response = model.generate(prompt="Tell me a joke", schema=Joke)
        assert issubclass(type(response), BaseModel)


def test_correctness_metric():
    test_case = LLMTestCase(
        input="The dog chased the cat up the tree, who ran up the tree?",
        actual_output="It depends, some might consider the cat, while others might argue the dog.",
        expected_output="The cat."
    )
    
    with (
        patch.object(
            correctness_metric, "_generate_evaluation_steps", return_value=["first step", "second step"]
        ),
        patch.object(
            correctness_metric,"evaluate", return_value=(1.0, "all good")
        ) as mocked_evaluate,
    ):
        correctness_metric.measure(test_case)
        mocked_evaluate.assert_called_with(test_case)
        assert isinstance(correctness_metric.score, float)
        assert isinstance(correctness_metric.reason, str)
