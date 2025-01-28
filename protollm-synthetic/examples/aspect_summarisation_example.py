from protollm_synthetic.synthetic_pipelines.chains import AspectSummarisationChain
from protollm_synthetic.utils import Dataset, VLLMChatOpenAI
import pandas as pd
import os
import asyncio

texts = [
    "The quick brown fox jumps over the lazy dog. The fox is a cunning animal as some politicians.",
    "Artificial intelligence is transforming the world. AI is a powerful technology and influence much on the world politics.",
    "Python is a popular programming language."
]

df = pd.DataFrame(texts, columns=["content"])
df.to_json("tmp_data/tmp_sample_summarization_dataset.json", index=False)

dataset = Dataset(path="tmp_data/tmp_sample_summarization_dataset.json")
# Expected output: a list of summaries
expected_summaries = [
    "The fox jumps over the dog.",
    "AI is changing the world.",
    "Python is a popular language."
]

aspect = "politics"

qwen_large_api_key = os.environ.get("QWEN2VL_OPENAI_API_KEY")
qwen_large_api_base = os.environ.get("QWEN2VL_OPENAI_API_BASE")

llm=VLLMChatOpenAI(
        api_key=qwen_large_api_key,
        base_url=qwen_large_api_base,
        model="/model",
        max_tokens=2048,
        # max_concurrency=10
    )

aspect_summarisation_chain = AspectSummarisationChain(llm=llm)
actual_summaries = asyncio.run(aspect_summarisation_chain.run(dataset, aspect=aspect, n_examples=3))
print(actual_summaries)