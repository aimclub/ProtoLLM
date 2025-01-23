from protollm_worker.config import MODEL_PATH, REDIS_HOST, REDIS_PORT, QUEUE_NAME
from protollm_worker.models.vllm_models import VllMModel
from protollm_worker.services.broker import LLMWrap
from protollm_worker.config import Config

if __name__ == "__main__":
    config = Config.read_from_env()
    llm_model = VllMModel(model_path=MODEL_PATH)
    llm_wrap = LLMWrap(llm_model=llm_model,
                       config= config)
    llm_wrap.start_connection()
