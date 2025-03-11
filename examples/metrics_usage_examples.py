# To use metrics, simply import the desired one from ProtoLLM or directly from deepeval. In the second case, you
# may also need to import a connector object for deepeval metrics to work. This can be done as follows:
#
# `from protollm.metrics.evaluation_metrics import model`
#
# Also make sure that you set the model URL and model_name in the same format as for a normal LLM connector
# (URL;model_name).
# Detailed documentation on metrics is available at the following URL:
# https://docs.confident-ai.com/docs/metrics-introduction

import logging

from deepeval.test_case import LLMTestCase, ToolCall

from protollm.metrics import answer_relevancy, tool_correctness

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Create test case for metric
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )

    answer_relevancy.measure(test_case) # Evaluate metric
    logging.info(f"Answer relevancy score {answer_relevancy.score}")
    logging.info(f"Answer relevancy reason: {answer_relevancy.reason}")
    
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
