from typing import Optional
from unittest.mock import patch

from deepeval.metrics.answer_relevancy.schema import AnswerRelevancyVerdict
from deepeval.metrics.contextual_precision.schema import ContextualPrecisionVerdict
from deepeval.metrics.contextual_recall.schema import ContextualRecallVerdict
from deepeval.metrics.contextual_relevancy.schema import ContextualRelevancyVerdicts
from deepeval.metrics.faithfulness.schema import FaithfulnessVerdict
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from protollm.metrics.deepeval_connector import DeepEvalConnector
from protollm.metrics.evaluation_metrics import (answer_relevancy,
                                                 context_precision,
                                                 context_recall,
                                                 context_relevancy,
                                                 correctness_metric,
                                                 faithfulness,
                                                 task_completion,
                                                 tool_correctness)


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


def test_answer_relevancy_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )
    
    with (
        patch.object(
            answer_relevancy, "_generate_statements", return_value=["first claim", "second claim"]
        ) as mocked_statements,
        patch.object(
            answer_relevancy,
            "_generate_verdicts",
            return_value=[AnswerRelevancyVerdict(**{"verdict": "yes"}), AnswerRelevancyVerdict(**{"verdict": "no"})]
        ) as mocked_verdicts,
        patch.object(
            answer_relevancy, "_generate_reason", return_value="all good"
        ) as mocked_reason
    ):
        answer_relevancy.measure(test_case)
        mocked_statements.assert_called_with(test_case.actual_output)
        mocked_verdicts.assert_called_with(test_case.input)
        mocked_reason.assert_called_with(test_case.input)
        assert isinstance(answer_relevancy.score, float)
        assert isinstance(answer_relevancy.reason, str)


def test_context_precision_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    
    with (
        patch.object(
            context_precision,
            "_generate_verdicts",
            return_value=[
                ContextualPrecisionVerdict(**{"verdict": "yes", "reason": "one"}),
                ContextualPrecisionVerdict(**{"verdict": "no", "reason": "two"})
            ]
        ) as mocked_verdicts,
        patch.object(
            context_precision, "_generate_reason", return_value="all good"
        ) as mocked_reason
    ):
        context_precision.measure(test_case)
        mocked_verdicts.assert_called_with(test_case.input, test_case.expected_output, test_case.retrieval_context)
        mocked_reason.assert_called_with(test_case.input)
        assert isinstance(context_precision.score, float)
        assert isinstance(context_precision.reason, str)


def test_context_recall_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    
    with (
        patch.object(
            context_recall,
            "_generate_verdicts",
            return_value=[
                ContextualRecallVerdict(**{"verdict": "yes", "reason": "one"}),
                ContextualRecallVerdict(**{"verdict": "no", "reason": "two"})
            ]
        ) as mocked_verdicts,
        patch.object(
            context_recall, "_generate_reason", return_value="all good"
        ) as mocked_reason
    ):
        context_recall.measure(test_case)
        mocked_verdicts.assert_called_with(test_case.expected_output, test_case.retrieval_context)
        mocked_reason.assert_called_with(test_case.expected_output)
        assert isinstance(context_recall.score, float)
        assert isinstance(context_recall.reason, str)


def test_context_relevancy_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    
    with (
        patch.object(
            context_relevancy,
            "_generate_verdicts",
            return_value=ContextualRelevancyVerdicts(
                **{
                    "verdicts":[
                        {"verdict": "yes", "statement": "one"},
                        {"verdict": "no", "statement": "second claim", "reason": "two"}
                    ]
                }
            )
        ) as mocked_verdicts,
        patch.object(
            context_relevancy, "_generate_reason", return_value="all good"
        ) as mocked_reason
    ):
        context_relevancy.measure(test_case)
        mocked_verdicts.assert_called_with(test_case.input, test_case.retrieval_context[0])
        mocked_reason.assert_called_with(test_case.input)
        assert isinstance(context_relevancy.score, float)
        assert isinstance(context_relevancy.reason, str)


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


def test_faithfulness_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    
    with (
        patch.object(
            faithfulness, "_generate_truths", return_value=["first truth", "second truth"]
        ) as mocked_truths,
        patch.object(
            faithfulness, "_generate_claims", return_value=["first claim", "second claim"]
        ) as mocked_claims,
        patch.object(
            faithfulness, "_generate_verdicts", return_value=[
                    FaithfulnessVerdict(**{"verdict": "yes", "reason": "one"}),
                    FaithfulnessVerdict(**{"verdict": "no", "reason": "two"})
            ]
        ),
        patch.object(
            faithfulness, "_generate_reason", return_value="all good"
        )
    ):
        faithfulness.measure(test_case)
        mocked_truths.assert_called_with(test_case.retrieval_context)
        mocked_claims.assert_called_with(test_case.actual_output)
        assert isinstance(faithfulness.score, float)
        assert isinstance(faithfulness.reason, str)
        

def test_task_completion_metric():
    test_case = LLMTestCase(
        input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
        actual_output=(
            "Day 1: Eiffel Tower, dinner at Le Jules Verne. "
            "Day 2: Louvre Museum, lunch at Angelina Paris. "
            "Day 3: Montmartre, evening at a wine bar."
        ),
        tools_called=[
            ToolCall(
                name="Itinerary Generator",
                description="Creates travel plans based on destination and duration.",
                input_parameters={"destination": "Paris", "days": 3},
                output=[
                    "Day 1: Eiffel Tower, Le Jules Verne.",
                    "Day 2: Louvre Museum, Angelina Paris.",
                    "Day 3: Montmartre, wine bar.",
                ],
            ),
            ToolCall(
                name="Restaurant Finder",
                description="Finds top restaurants in a city.",
                input_parameters={"city": "Paris"},
                output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
            ),
        ],
    )
    
    with (
        patch.object(
            task_completion, "_extract_goal_and_outcome", return_value=("goal", "outcome")
        ) as mocked_goal_and_outcome,
        patch.object(
            task_completion, "_generate_verdicts", return_value=(0.85, "just because")
        )
    ):
        task_completion.measure(test_case)
        mocked_goal_and_outcome.assert_called_with(test_case)
        assert isinstance(task_completion.score, float)
        assert isinstance(task_completion.reason, str)


def test_tool_correctness_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        tools_called=[ToolCall(name="WebSearch", input_parameters={}), ToolCall(name="ToolQuery", input_parameters={})],
        expected_tools=[ToolCall(name="WebSearch", input_parameters={})],
    )

    tool_correctness.measure(test_case)
    assert isinstance(tool_correctness.score, float)
    assert isinstance(tool_correctness.reason, str)
