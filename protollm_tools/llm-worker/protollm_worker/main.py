from protollm_worker.models.mock_llm import RandomLLM
from protollm_worker.services.broker import LLMWrap
from protollm_worker.config import Config

if __name__ == "__main__":

    config = Config()
    # config.queue_name = "..."
    llm_model = RandomLLM(
        name="random_llm_1",
        description="api_testing_model",
        ip_address="0.0.0.0"
    )
    llm_wrap = LLMWrap(
        llm_model=llm_model,
        config=config
    )
    llm_wrap.start_connection()
