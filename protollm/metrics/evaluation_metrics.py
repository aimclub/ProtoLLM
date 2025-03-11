from deepeval.metrics import (GEval,
                              AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric,
                              TaskCompletionMetric,
                              ToolCorrectnessMetric)
from deepeval.test_case import LLMTestCaseParams

from protollm.metrics.deepeval_connector import DeepEvalConnector

model_for_metrics = DeepEvalConnector()
metrics_init_params = {
    "model": model_for_metrics,
    "async_mode": False,
}

# Use LLMs
correctness_metric = GEval(
    name="Correctness",
    criteria=( # Сan be overridden for a specific task
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
answer_relevancy = AnswerRelevancyMetric(**metrics_init_params)
faithfulness = FaithfulnessMetric(**metrics_init_params)
context_precision = ContextualPrecisionMetric(**metrics_init_params)
context_recall = ContextualRecallMetric(**metrics_init_params)
context_relevancy = ContextualRelevancyMetric(**metrics_init_params)
task_completion = TaskCompletionMetric(**metrics_init_params)

# Don't use LLMs
tool_correctness = ToolCorrectnessMetric()
