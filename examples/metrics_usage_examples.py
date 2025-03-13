# You can use correctness_metric from ProtoLLM or import desired directly from deepeval.
# The correctness metric is demonstrative, but has been shown to be good for determining the correctness of an answer.
# You can define a new one, with a different criterion, if necessary.
#
# In the second case, you may also need to import a connector object for deepeval metrics to work. This can be done
# as follows:
#
# `from protollm.metrics import model_for_metrics`
#
# Also make sure that you set the model URL and model_name in the same format as for a normal LLM connector
# (URL;model_name).
#
# Detailed documentation on metrics is available at the following URL:
# https://docs.confident-ai.com/docs/metrics-introduction

import logging

from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from protollm.metrics import correctness_metric, model_for_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

answer_relevancy = AnswerRelevancyMetric(model=model_for_metrics, async_mode=False)
tool_correctness = ToolCorrectnessMetric()

if __name__ == "__main__":
    # ===================================metrics using LLM=============================================
    # Create test case for metric
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost."
    )

    answer_relevancy.measure(test_case) # Evaluate metric
    logging.info(f"Answer relevancy score {answer_relevancy.score}")
    logging.info(f"Answer relevancy reason: {answer_relevancy.reason}")
    
    correctness_metric.measure(test_case) # Evaluate metric
    logging.info(f"Correctness score {correctness_metric.score}")
    logging.info(f"Correctness reason: {correctness_metric.reason}")
    
    # ===================================metrics not using LLM=========================================
    # Create test case for metric
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name="WebSearch", input_parameters={}), ToolCall(name="ToolQuery", input_parameters={})],
        expected_tools=[ToolCall(name="WebSearch", input_parameters={})],
    )
    
    tool_correctness.measure(test_case)
    logging.info(f"Tool correctness score {tool_correctness.score}")
    logging.info(f"Tool correctness reason: {tool_correctness.reason}")
