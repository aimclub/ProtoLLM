import warnings
from typing import Any, Optional, Callable, Dict

from langchain_core.documents import Document

from langchain_chroma import Chroma

from dataclasses import dataclass

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from chromadb import ClientAPI


@dataclass
class DocsSearcherModels:
    embedding_model: SentenceTransformerEmbeddings | None | Any = None
    chroma_client: ClientAPI | None = None


class DocRetriever:
    def __init__(
        self,
        top_k: int,
        docs_searcher_models: DocsSearcherModels,
        preprocess_query: Optional[Callable[[str], str]] = None,
    ):
        """
        :param preprocess_query: for example, get keywords from query
        """
        self.top_k = top_k
        self.embedding_function = docs_searcher_models.embedding_model
        self.client = docs_searcher_models.chroma_client
        self.preprocess_query = preprocess_query

    def retrieve_top(
        self, collection_name: str, query: str, filter: Optional[Dict[str, Any]] = None
    ) -> Optional[list[Document]]:
        """
        Retrieve top K documents from Vector DB (e.x., Chroma).
        """
        if filter is None:
            _filter = {}
        else:
            _filter = filter.copy()
        if collection_name is None:
            warnings.warn("Collection name is None")
            return None
        if self.preprocess_query is not None:
            query = self.preprocess_query(query)
        if collection_name not in [col.name for col in self.client.list_collections()]:
            warnings.warn(
                "There is no collection named {} in Chroma DB".format(collection_name)
            )
            return None

        store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )
        return store.as_retriever(
            search_kwargs={"k": self.top_k, "filter": _filter}
        ).invoke(query)


class RetrievingPipeline:
    def __init__(self):
        self._retrievers: Optional[list[DocRetriever]] = None
        self._collection_names: Optional[list[str]] = None

    def set_retrievers(self, retrievers: list[DocRetriever]) -> "RetrievingPipeline":
        self._retrievers = retrievers
        return self

    def set_collection_names(self, collection_names: list[str]) -> "RetrievingPipeline":
        self._collection_names = collection_names
        return self

    def get_retrieved_docs(self, query: str) -> list[Document]:
        if any([self._retrievers is None, self._collection_names is None]):
            raise ValueError("Either retrievers or collection_names must not be None")

        if len(self._retrievers) == len(self._collection_names):
            _query = query
            docs = self._retrievers[0].retrieve_top(self._collection_names[0], _query)
            for i in range(1, len(self._retrievers)):
                filter = {"uuid": {"$in": [doc.metadata["uuid"] for doc in docs]}}
                docs_next = self._retrievers[i].retrieve_top(
                    self._collection_names[i], _query, filter
                )
                docs = docs_next
        else:
            raise Exception("The length of retrievers and collection_names must match")

        return docs
