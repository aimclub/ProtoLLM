from copy import deepcopy

import chromadb
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from protollm.rags.rag_core.retriever import DocRetriever, DocsSearcherModels
from protollm.rags.settings.chroma_settings import (
    ChromaSettings,
    settings as default_settings,
)
from protollm.rags.stores.chroma.chroma_loader import load_documents_to_chroma_db


def chroma_loading(path: str, collection: str) -> None:
    # Loads data to ChromaDB

    default_settings.collection_name = collection
    default_settings.docs_collection_path = path
    processing_batch_size = 32
    loading_batch_size = 32
    settings = deepcopy(default_settings)
    load_documents_to_chroma_db(
        settings=settings,
        processing_batch_size=processing_batch_size,
        loading_batch_size=loading_batch_size,
    )


def chroma_view(
    query: str,
    collection: str,
    k: int = 1,
    embedding_function: HuggingFaceHubEmbeddings = None,
) -> list:
    # Returns k chunks that are closest to the query
    chroma_client = init_chroma_client()
    embedding_function = HuggingFaceHubEmbeddings(model=default_settings.embedding_host)
    default_settings.collection_name = collection

    chroma_collection = Chroma(
        collection_name=collection,
        embedding_function=embedding_function,
        client=chroma_client,
    )

    return chroma_collection.similarity_search_with_score(query, k)


def init_chroma_client():
    return chromadb.HttpClient(
        host=default_settings.chroma_host,
        port=default_settings.chroma_port,
        settings=chromadb.Settings(allow_reset=default_settings.allow_reset),
    )


def proto_view(
    query: str,
    collection: str,
    k: int = 1,
) -> list:
    # Returns k chunks that are closest to the query
    embedding_function = HuggingFaceHubEmbeddings(model=default_settings.embedding_host)
    chroma_client = init_chroma_client()

    docs_searcher_models = DocsSearcherModels(
        embedding_model=embedding_function, chroma_client=chroma_client
    )
    retriever = DocRetriever(
        top_k=k,
        docs_searcher_models=docs_searcher_models,
    )

    return retriever.retrieve_top(collection_name=collection, query=query)


def delete_collection(collection: str) -> None:
    # Deletes the collection
    chroma_client = init_chroma_client()
    chroma_client.delete_collection(collection)


def list_collections() -> list:
    # Returns a list of all collections
    chroma_client = init_chroma_client()
    return chroma_client.list_collections()


def create_base_collection(
    collection_name: str,
    docs_collection: list[str],
) -> chromadb.Collection:
    """Creates a new collection in the ChromaDB and adds the given documents to it.

    Args:
        collection_name (str): The name of the new collection.
        docs_collection (list[str]): The list of documents to add to the collection.
        Must be just a list of any strings, which should not be too long due to vectorization.

    Returns:
        chromadb.Collection: The newly created collection.
    """
    chroma_client = init_chroma_client()
    embedding_function = HuggingFaceHubEmbeddings(model=default_settings.embedding_host)
    collection = chroma_client.create_collection(
        collection_name,
        embedding_function=embedding_function,
    )
    collection.add(
        documents=docs_collection,
        ids=[str(i) for i in range(len(docs_collection))],
    )
    return collection


if __name__ == "__main__":
    collection_name = "external-document"
    query = "Как называется документ?"
    res = proto_view(query, collection_name)
    print(res[0][0].page_content)
