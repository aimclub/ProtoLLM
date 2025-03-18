import chromadb
import os
import uuid
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings

from protollm_sdk.models.job_context_models import PromptModel
from protollm_sdk.jobs.outer_llm_api import OuterLLMAPI
from protollm.rags.rag_core.retriever import DocRetriever, DocsSearcherModels

from protollm.definitions import CONFIG_PATH


def init_chroma_client():
    host, port = os.environ.get("CHROMA_DEFAULT_SETTINGS").split(':')
    return chromadb.HttpClient(
        host=host,
        port=int(port),
        settings=chromadb.Settings(),
    )


def proto_view(
    query: str,
    collection: str,
    k: int = 1,
    embedding_function: HuggingFaceHubEmbeddings = None,
) -> list:
    # Returns k chunks that are closest to the query
    embedding_host = os.environ.get("EMBEDDING_HOST")
    embedding_function = HuggingFaceHubEmbeddings(model=embedding_host)
    chroma_client = init_chroma_client()

    docs_searcher_models = DocsSearcherModels(embedding_model=embedding_function, chroma_client=chroma_client)
    retriever = DocRetriever(top_k=k,
                             docs_searcher_models=docs_searcher_models,
                             )

    return retriever.retrieve_top(collection_name=collection, query=query)


def outer_llm(question: str,
              meta: dict,
              key: str):
    llmapi = OuterLLMAPI(key)
    llm_request = PromptModel(job_id=str(uuid.uuid4()),
                              meta=meta,
                              content=question)
    res = llmapi.inference(llm_request)
    return res.content


if __name__ == "__main__":
    load_dotenv(CONFIG_PATH)

    # Настройки БЯМ
    meta = {"temperature": 0.05,
            "tokens_limit": 4096,
            "stop_words": None}
    key = os.environ.get("VSE_GPT_KEY")


    # Название коллекции в БД
    collection_name = "strategy-spb"

    # Вопрос
    question = 'Какие задачи Стратегия ставит в области энергосбережения?'

    # Извлечение контекста из БД
    context = proto_view(question, collection_name)
    context = f'Вопрос: {question} Контекст: {context[0].page_content}'

    # Получение ответа от БЯМ
    print(f'Ответ VseGPT LLM: \n {outer_llm(context, meta, key)}')
