import json
import os
import chromadb
import numpy as np
import pytest
from fastapi.testclient import TestClient
import psycopg2
from protollm_agents.entrypoint import Entrypoint
from protollm_agents.sdk.models import (
    CompletionModel,
    ChatModel,
    TokenizerModel,
    EmbeddingAPIModel,
)


@pytest.fixture(scope="session")
def docker_compose_command() -> str:
    return "docker compose"


# @pytest.fixture(scope="session")
# def docker_setup():
#     pass

# @pytest.fixture(scope="session")
# def docker_cleanup():
#     pass


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    return "tests/docker-compose.test.yml"


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    return "test-protollm-agents"


@pytest.fixture(scope="session")
def test_config_path():
    return "tests/config.test.yml"


def is_responsive_db(host, port):
    try:
        conn = psycopg2.connect(
            host=host, port=port, user="test", password="test", dbname="test"
        )
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False


def is_responsive_vectorstore(host, port):
    try:
        chroma_client = chromadb.HttpClient(host=host, port=port)
        chroma_client.list_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def test_client(test_config_path, docker_services, docker_ip):
    port_db = docker_services.port_for("db", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_db(host=docker_ip, port=port_db),
    )
    port_vectorstore = docker_services.port_for("vectorstore", 8000)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_vectorstore(host=docker_ip, port=port_vectorstore),
    )
    models = [
        CompletionModel(
            name="planner_llm",
            model="/model",
            temperature=0.01,
            top_p=0.95,
            streaming=False,
            url=f"http://{os.getenv('PYTEST_COMPLETION_MODEL_HOST')}:{os.getenv('PYTEST_COMPLETION_MODEL_PORT')}/v1",
            api_key=os.getenv("PYTEST_COMPLETION_MODEL_API_KEY"),
        ),
        CompletionModel(
            name="generator_llm",
            model="/model",
            temperature=0.01,
            top_p=0.95,
            streaming=True,
            url=f"http://{os.getenv('PYTEST_COMPLETION_MODEL_HOST')}:{os.getenv('PYTEST_COMPLETION_MODEL_PORT')}/v1",
            api_key=os.getenv("PYTEST_COMPLETION_MODEL_API_KEY"),
        ),
        ChatModel(
            name="router_llm",
            model="/model",
            temperature=0.01,
            top_p=0.95,
            streaming=True,
            url=f"http://{os.getenv('PYTEST_CHAT_MODEL_HOST')}:{os.getenv('PYTEST_CHAT_MODEL_PORT')}/v1",
            api_key=os.getenv("PYTEST_CHAT_MODEL_API_KEY"),
        ),
        TokenizerModel(
            name="qwen_2.5",
            path_or_repo_id="Qwen/Qwen2.5-7B-Instruct",
        ),
        EmbeddingAPIModel(
            name="e5-mistral-7b-instruct",
            model="/models/e5-mistral-7b-instruct",
            url=f"http://{os.getenv('PYTEST_EMBEDDING_MODEL_HOST')}:{os.getenv('PYTEST_EMBEDDING_MODEL_PORT')}/v1",
            api_key=os.getenv("PYTEST_EMBEDDING_MODEL_API_KEY"),
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        ),
    ]

    entrypoint = Entrypoint(config_path=test_config_path, models=models)
    with TestClient(entrypoint.app) as client:
        yield client


@pytest.fixture(scope="function")
def create_collections(request, docker_services, docker_ip):
    port = docker_services.port_for("vectorstore", 8000)
    collection_datas = {
        "test_collection_1": "col_data_ed.json",
        "test_collection_2": "col_data_env.json",
        "test_collection_3": None,
    }
    chroma_client = chromadb.HttpClient(host=docker_ip, port=port)
    for collection_name, collection_data_path in collection_datas.items():
        if collection_name in [col.name for col in chroma_client.list_collections()]:
            chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(name=collection_name)
        if collection_data_path is None:
            ids, documents, embeddings, metadatas = [], [], [], []
            for path in collection_datas.values():
                if path is not None:
                    with open(
                        os.path.join(request.config.rootdir, "tests", "docs", path), "r"
                    ) as f:
                        data = json.load(f)
                        ids.extend(data.get("ids"))
                        documents.extend(data.get("documents"))
                        embeddings.extend(data.get("embeddings"))
                        metadatas.extend(data.get("metadatas"))
        else:
            with open(
                os.path.join(
                    request.config.rootdir, "tests", "docs", collection_data_path
                ),
                "r",
            ) as f:
                collection_data = json.load(f)
            ids = collection_data.get("ids")
            documents = collection_data.get("documents")
            embeddings = np.array(collection_data.get("embeddings"))
            metadatas = collection_data.get("metadatas")
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    yield collection_datas.keys()
    for collection_name in collection_datas.keys():
        chroma_client.delete_collection(name=collection_name)
