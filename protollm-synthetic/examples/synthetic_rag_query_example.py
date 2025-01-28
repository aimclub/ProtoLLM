import logging
import os
from protollm_synthetic.synthetic_pipelines.chains import RAGChain
from protollm_synthetic.utils import Dataset, VLLMChatOpenAI
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# path = 'tmp_data/sample_data_city_rag.json'
path = 'tmp_data/sample_data_rag_spb.json'
dataset = Dataset(data_col='content', path=path)

qwen_large_api_key = os.environ.get("QWEN_OPENAI_API_KEY")
qwen_large_api_base = os.environ.get("QWEN_OPENAI_API_BASE")

logger.info("Initializing LLM connection")

llm=VLLMChatOpenAI(
        api_key=qwen_large_api_key,
        base_url=qwen_large_api_base,
        model="/model",
        max_tokens=2048,
    )

rag_chain = RAGChain(llm=llm)

logger.info("Starting generating")
asyncio.run(rag_chain.run(dataset, 
                          n_examples=5))

logger.info("Saving results")
path = 'tmp_data/sample_data_city_rag_generated.json'

# An alternative way to save data
# rag_chain.save_chain_output('tmp_data/sample_data_city_rag_generated.json')

df = rag_chain.data.explode('generated')
df['question'] = df['generated'].apply(lambda x: x['question'])
df['answer'] = df['generated'].apply(lambda x: x['answer'])
df = df[['content', 'question', 'answer']]

logger.info(f"Writing result to {path}")
df.to_json(path, orient="records")

logger.info("Generation successfully finished")
