from langchain_community.vectorstores import VectorStore, Chroma
import chromadb
from pydantic import Field

from langchain_community.vectorstores import VectorStore

from .base import BaseVectorStore 


class ChromaVectorStore(BaseVectorStore):
    host: str = Field(..., description="Host of the vector store")
    port: int = Field(..., description="Port of the vector store")
    collection_name: str = Field(..., description="Collection name of the vector store")

    def to_vector_store(self) -> VectorStore:
        if self.embeddings_model is None:
            raise ValueError("Embeddings model is not initialized")
        return Chroma(
            client=chromadb.HttpClient(host=self.host, port=self.port),
            collection_name=self.collection_name,
            embedding_function=self.embeddings_model
        )


