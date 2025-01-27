from protollm_worker.models.vllm_models import VllMModel
from protollm_worker.services.broker import LLMWrap
from protollm_worker.config import Config

if __name__ == "__main__":
    config = Config.read_from_env()
    llm_model = VllMModel(model_path=config.model_path,
                          tensor_parallel_size=config.tensor_parallel_size,
                          gpu_memory_utilisation=config.gpu_memory_utilisation,
                          tokens_len=config.token_len)
    llm_wrap = LLMWrap(llm_model=llm_model,
                       config= config)
    llm_wrap.start_connection()
