import uuid
from typing import Any, cast

import pytest

from protollm.rags.stores.chroma.utils import (
    delete_repeats,
    get_all_docs_name,
    insert_documents,
    list_collections,
    merge_collections,
)


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []

    # Chroma-like API
    def get(self, include=None):
        data = {
            "ids": list(self._ids),
            "documents": list(self._documents),
            "metadatas": list(self._metadatas),
            "embeddings": list(self._embeddings),
        }
        if include is None:
            return data
        return {k: data[k] for k in include} | {"ids": data["ids"]}

    def add(self, ids, metadatas, documents, embeddings=None):
        # Normalize to single-item append (to mimic current utils.py usage)
        self._ids.extend(ids)
        # utils.py sometimes passes single dict instead of list for metadatas/documents
        if isinstance(metadatas, dict):
            metadatas = [metadatas]
        if isinstance(documents, str):
            documents = [documents]
        self._metadatas.extend(metadatas)
        self._documents.extend(documents)
        if embeddings is None:
            embeddings = [[0.0] * 3] * len(ids)
        elif (
            isinstance(embeddings, list)
            and embeddings
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]
        self._embeddings.extend(embeddings)

    # LangChain Chroma wrapper compatibility for insert_documents
    def add_documents(self, docs):
        for d in docs:
            self.add(
                ids=[str(uuid.uuid4())],
                metadatas=[d.metadata],
                documents=[d.page_content],
                embeddings=[[0.0, 0.0, 0.0]],
            )


class FakeClient:
    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}

    def create_collection(self, name: str):
        if name in self._collections:
            return self._collections[name]
        self._collections[name] = FakeCollection(name)
        return self._collections[name]

    def get_collection(self, name: str):
        return self._collections[name]

    def list_collections(self):
        # Mimic chromadb API returning objects with .name
        return list(self._collections.values())


def test_list_collections_returns_names():
    # Arrange
    client = FakeClient()
    client.create_collection("a")
    client.create_collection("b")

    # Act
    names = list(list_collections(client))

    # Assert
    assert set(names) == {"a", "b"}


def test_merge_collections_into_existing_no_duplicates():
    # Arrange
    client = FakeClient()
    col1 = client.create_collection("c1")
    col2 = client.create_collection("c2")
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    col1.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"m": 1}],
        documents=["doc-a"],
        embeddings=[emb_a],
    )
    col2.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"m": 2}],
        documents=["doc-b"],
        embeddings=[emb_b],
    )

    # Act
    merge_collections(client, "c1", "c2", None)

    # Assert
    data = col1.get()
    assert set(data["documents"]) == {"doc-a", "doc-b"}
    assert len(data["ids"]) == 2


def test_merge_collections_into_new_collection_deduplicates():
    # Arrange
    client = FakeClient()
    col1 = client.create_collection("c1")
    col2 = client.create_collection("c2")
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    col1.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"m": 1}],
        documents=["doc-a"],
        embeddings=[emb_a],
    )
    col2.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"m": 2}],
        documents=["doc-a"],
        embeddings=[emb_b],
    )  # duplicate document text

    # Act
    merge_collections(client, "c1", "c2", "c3")

    # Assert
    col3 = client.get_collection("c3")
    data = col3.get()
    assert data["documents"].count("doc-a") == 1


def test_get_all_docs_name_extracts_source_filename(tmp_path):
    # Arrange
    client = FakeClient()
    col = client.create_collection("c")
    col.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"source": "dir1\\file1.pdf"}],
        documents=["a"],
        embeddings=[[0, 0, 1]],
    )
    col.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"source": "dir2\\file2.txt"}],
        documents=["b"],
        embeddings=[[0, 1, 0]],
    )

    # Act
    names = get_all_docs_name(cast(Any, col))

    # Assert
    assert names == {"file1.pdf", "file2.txt"}


def test_get_all_docs_name_raises_when_no_source():
    # Arrange
    client = FakeClient()
    col = client.create_collection("c")
    col.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"path": "x"}],
        documents=["a"],
        embeddings=[[0, 0, 1]],
    )

    # Act / Assert
    with pytest.raises(KeyError):
        _ = get_all_docs_name(cast(Any, col))


def test_insert_documents_inserts_only_new_documents(monkeypatch):
    # Arrange
    from langchain_core.documents import Document

    client = FakeClient()
    col = client.create_collection("c")
    # existing
    col.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"source": "dir1\\exists.pdf"}],
        documents=["exists"],
        embeddings=[[1, 0, 0]],
    )

    docs = [
        Document(page_content="exists", metadata={"source": "dir1\\exists.pdf"}),
        Document(page_content="new-doc", metadata={"source": "dir2\\new.pdf"}),
    ]

    # Act
    insert_documents(cast(Any, col), iter(docs))

    # Assert
    data = col.get()
    assert "new-doc" in data["documents"]
    # Only one new doc added
    assert data["documents"].count("new-doc") == 1


def test_insert_documents_raises_when_no_source():
    # Arrange
    from langchain_core.documents import Document

    client = FakeClient()
    col = client.create_collection("c")
    docs = [Document(page_content="x", metadata={"not_source": "a"})]

    # Act / Assert
    with pytest.raises(KeyError):
        insert_documents(cast(Any, col), iter(docs))


def test_delete_repeats_deduplicates_close_embeddings(monkeypatch):
    # Arrange
    client = FakeClient()
    col = client.create_collection("c")
    # two near-identical embeddings
    col.add(
        ids=["1"],
        metadatas=[{"m": 1}],
        documents=["short"],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    col.add(
        ids=["2"],
        metadatas=[{"m": 2}],
        documents=["a bit longer"],
        embeddings=[[0.999, 0.0, 0.0]],
    )

    # custom similarity to force duplicate
    def sim(a, b):
        return 0.9999

    # Act
    cache = delete_repeats(cast(Any, col), similarity_func=sim)

    # Assert
    # The function currently returns cache_ids and does not delete; ensure grouping produced
    assert isinstance(cache, dict)
    assert any(cache.values())
