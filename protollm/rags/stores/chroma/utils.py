import uuid
from collections import defaultdict
from typing import Any, Iterable, Callable

import chromadb
import numpy as np
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm


def merge_collections(
    chroma_client: chromadb.HttpClient,
    collection_name_1: str,
    collection_name_2: str,
    new_collection_name: str | None = None,
):
    """
    Merge 2 collections into the 'collection_name_1' if 'new_collection_name' is None,
    otherwise merge 2 collections into the new one with 'new_collection_name' name.
    If any problems with network or DB accessibility will occur, exception are raised

    :raises Exception: if there are network or database accessibility issues.
    """

    collection_1 = chroma_client.get_collection(name=collection_name_1)
    collection_2 = chroma_client.get_collection(name=collection_name_2)

    docs_1: dict[str, Any] = collection_1.get(
        include=["documents", "metadatas", "embeddings"]
    )
    docs_2: dict[str, Any] = collection_2.get(
        include=["documents", "metadatas", "embeddings"]
    )

    if new_collection_name is None:
        for i in range(len(docs_2["ids"])):
            if docs_2["documents"][i] not in docs_1["documents"]:
                collection_1.add(
                    ids=[str(uuid.uuid4())],
                    metadatas=[docs_2["metadatas"][i]],
                    documents=[docs_2["documents"][i]],
                    embeddings=[docs_2["embeddings"][i]],
                )
        return

    new_collection = chroma_client.create_collection(name=new_collection_name)

    merged_docs = docs_1
    for i in range(len(docs_2["ids"])):
        if docs_2["documents"][i] not in merged_docs["documents"]:
            merged_docs["ids"].append(str(uuid.uuid4()))
            merged_docs["embeddings"].append(docs_2["embeddings"][i])
            merged_docs["metadatas"].append(docs_2["metadatas"][i])
            merged_docs["documents"].append(docs_2["documents"][i])

    for i in range(len(merged_docs["ids"])):
        new_collection.add(
            ids=[str(uuid.uuid4())],
            metadatas=merged_docs["metadatas"][i],
            documents=merged_docs["documents"][i],
        )


def delete_repeats(
    collection: Chroma,
    similarity_func: Callable[[list[float], list[float]], float] = None,
) -> None:
    """
    Remove duplicate documents from a collection.

    :param collection: it should include fields: 'documents', 'embeddings', 'metadatas'

    :raises Exception: if there are issues accessing the database.
    """

    def cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )

    if similarity_func is None:
        similarity_func = cosine_similarity
    docs = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = docs["ids"]
    documents = docs["documents"]
    embeddings = docs["embeddings"]
    delete_ids = []
    cache_ids = defaultdict(list)

    for i in tqdm(range(len(docs["ids"]))):
        for j in range(i):
            if j in cache_ids[i]:
                continue
            if similarity_func(embeddings[i], embeddings[j]) > 0.97:
                if cache_ids[j]:
                    cache_ids[i].extend(cache_ids[j])
                    break
                cache_ids[i].append(j)
        delete_ids.extend(
            [
                ids[j] if len(documents[j]) > len(documents[i]) else ids[i]
                for j in cache_ids[i]
            ]
        )

    return cache_ids
    # delete_ids = list(set(delete_ids))
    # collection.delete(ids=delete_ids)


def list_collections(client: chromadb.HttpClient) -> Iterable[str]:
    return [collection.name for collection in client.list_collections()]


def get_all_docs_name(collection: Chroma) -> set[str]:
    """
    Return list of files' name from collection.

    :param collection: it should include fields: 'documents', 'embeddings', 'metadatas'

    :raises KeyError: if there is no key 'source' in the documents' metadata from 'collection'
    """
    docs: dict[str, Any] = collection.get()

    if "source" not in docs["metadatas"][0].keys():
        raise KeyError("There is no file name, called <source>, in document metadata")

    return set(
        str(metadata["source"].split("\\")[-1]) for metadata in docs["metadatas"]
    )


def insert_documents(collection: Chroma, docs: Iterable[Document]):
    """
    Insert only documents whose names aren't in 'collection'

    :raises KeyError: if there is no key 'source' in the documents' metadata from 'collection' or 'docs'
    """
    first_element = next(docs)
    if "source" not in first_element.metadata.keys():
        raise KeyError("There is no file name, called <source>, in document metadata")

    existing_docs_name = set(get_all_docs_name(collection))
    new_docs_name = set(
        [str(doc.metadata["source"].split("\\")[-1]) for doc in docs]
        + [str(first_element.metadata["source"].split("\\")[-1])]
    )
    docs_name_for_insert = new_docs_name.difference(existing_docs_name)
    if first_element.metadata["source"].split("\\")[-1] in docs_name_for_insert:
        docs_for_insert = [first_element]
    else:
        docs_for_insert = []
    docs_for_insert += [
        doc
        for doc in docs
        if doc.metadata["source"].split("\\")[-1] in docs_name_for_insert
    ]
    if docs_for_insert:
        collection.add_documents(docs_for_insert)
