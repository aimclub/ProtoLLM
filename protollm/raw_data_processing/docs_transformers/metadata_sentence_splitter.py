import logging
import re
from typing import Any, Iterable, Optional, List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from protollm.raw_data_processing.docs_transformers.utilities import (
    fix_list_dots_separators,
)

logger = logging.getLogger(__name__)


class DivMetadataSentencesSplitter(TextSplitter):
    """
    Splitter that divide document into sentences and add the source of this sentence to metadata.

    Example:
        Document:
        - page_content = "There is a huge number of clustering algorithms. The main idea of most of them is to combine
            identical sequences into one class or cluster based on similarity.
            As a rule, the choice of algorithm is determined by the task at hand."
        - metadata = {...}

        After splitting there are 3 Documents:
        [
            Document_1:
                - page_content = "There is a huge number of clustering algorithms."
                - metadata = {the same of Document, 'div': 'There is a huge number of clustering algorithms.
                    The main idea of most of them is to combine identical sequences into one class or cluster based on similarity.
                    As a rule, the choice of algorithm is determined by the task at hand.'}
            Document_2:
                - page_content = "The main idea of most of them is to combine identical sequences into one class or cluster based on similarity."
                - metadata = {the same of Document, 'div': 'There is a huge number of clustering algorithms.
                    The main idea of most of them is to combine identical sequences into one class or cluster based on similarity.
                    As a rule, the choice of algorithm is determined by the task at hand.'}
            Document_3:
                - page_content = "As a rule, the choice of algorithm is determined by the task at hand.",
                - metadata = {the same of Document, 'div': 'There is a huge number of clustering algorithms.
                    The main idea of most of them is to combine identical sequences into one class or cluster based on similarity.
                    As a rule, the choice of algorithm is determined by the task at hand.'}
        ]
    """

    def __init__(self, separators: Optional[Iterable[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._separators = separators or [r"\. (?<![0-9]\. | ^[0-9].)"]

    def _create_document(self, text: str, metadata: dict[str, Any]) -> Document:
        text_len = self._length_function(text)
        if text_len > self._chunk_size:
            logger.warning(
                f"A chunk of size {text_len} was encountered, "
                f"which is larger than the specified {self._chunk_size}"
            )
        return Document(page_content=text, metadata=metadata)

    def _split_on_sentences_with_additional_metadata(
        self, documents: Iterable[Document]
    ) -> List[Document]:
        splitted_docs = []
        for doc in documents:
            doc_metadata = doc.metadata
            doc_metadata["div"] = doc.page_content
            splitted_docs.extend(
                [
                    self._create_document(text, doc_metadata)
                    for text in self.split_text(doc.page_content)
                ]
            )

        return splitted_docs

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        return self._split_on_sentences_with_additional_metadata(documents)

    def split_text(self, text: str) -> List[str]:
        sentences_lst = [
            x.strip() for x in re.split(" | ".join(self._separators), text)
        ]
        return fix_list_dots_separators(sentences_lst)
