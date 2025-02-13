from protollm_worker.models.open_api_llm import OpenAPILLM
from protollm_worker.services.broker import LLMWrap
from protollm_worker.config import Config

if __name__ == "__main__":
    config = Config.read_from_env()
    llm_model = OpenAPILLM(model_url="https://api.vsegpt.ru/v1",
                           token="sk-or-vv-7fcc4ab944ca013feb7608fb7c0f001e5c12c32abf66233aad414183b4191a79",
                           default_model="openai/gpt-4o-2024-08-06",
                           # app_tag="test_protollm_worker"
                           )
    # llm_model = VllMModel(model_path=config.model_path,
    #                       tensor_parallel_size=config.tensor_parallel_size,
    #                       gpu_memory_utilisation=config.gpu_memory_utilisation,
    #                       tokens_len=config.token_len)
    llm_wrap = LLMWrap(llm_model=llm_model,
                       config= config)
    llm_wrap.start_connection()
