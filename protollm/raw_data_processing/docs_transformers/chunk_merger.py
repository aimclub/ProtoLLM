import logging
from typing import Any, Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

logger = logging.getLogger(__name__)


def _get_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    chapter = metadata["headings"][0] if metadata.get("headings", []) else ""
    source = metadata.get("source", "")
    meta = {
        "chapter": chapter,
        "source": source,
        "file_name": metadata.get("file_name", ""),
        "keywords": metadata.get("keywords", []),
        "object": [metadata.get("object", "")],
        "action": [metadata.get("action", "")],
    }
    return meta


class ChunkMerger(TextSplitter):
    def __init__(self, joiner: str = " ", **kwargs: Any):
        super().__init__(**kwargs)
        self.joiner = joiner

    def _merge_documents(self, documents: Iterable[Document]) -> List[Document]:
        documents = list(documents)
        if not documents:
            return []
        first_doc = documents[0]
        merged_chunk_content = first_doc.page_content
        merged_chunk_meta = _get_metadata(first_doc.metadata)
        chapter, source = merged_chunk_meta["chapter"], merged_chunk_meta["source"]
        transformed_docs = []
        for doc in documents[1:]:
            doc_text = doc.page_content
            doc_meta = _get_metadata(doc.metadata)
            doc_chapter, doc_source = doc_meta["chapter"], doc_meta["source"]

            if doc_chapter == chapter and doc_source == source:
                merged_text = self.joiner.join((merged_chunk_content, doc_text))
                if self._length_function(merged_text) <= self._chunk_size:
                    merged_chunk_content = merged_text
                    merged_chunk_meta["keywords"] += doc_meta["keywords"]
                    merged_chunk_meta["object"] += doc_meta["object"]
                    merged_chunk_meta["action"] += doc_meta["action"]
                    continue

            if merged_chunk_content:
                merged_chunk_meta["keywords"] = ", ".join(
                    set(merged_chunk_meta["keywords"])
                )
                merged_chunk_meta["object"] = ", ".join(
                    set(merged_chunk_meta["object"])
                )
                merged_chunk_meta["action"] = ", ".join(
                    set(merged_chunk_meta["action"])
                )
                transformed_docs.append(
                    self._create_document(merged_chunk_content, merged_chunk_meta)
                )

            merged_chunk_content, merged_chunk_meta = doc_text, doc_meta
            chapter, source = doc_chapter, doc_source

        if merged_chunk_content:
            transformed_docs.append(
                self._create_document(merged_chunk_content, merged_chunk_meta)
            )

        return transformed_docs

    def _create_document(self, text: str, metadata: dict[str, Any]) -> Document:
        text_len = self._length_function(text)
        if text_len > self._chunk_size:
            logger.warning(
                f"A chunk of size {text_len} was encountered, "
                f"which is larger than the specified {self._chunk_size}"
            )
        return Document(page_content=text, metadata=metadata)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        return self._merge_documents(documents)

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("split_text is not implemented in ChunkMerger")

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        raise NotImplementedError("_merge_splits is not implemented in ChunkMerger")
