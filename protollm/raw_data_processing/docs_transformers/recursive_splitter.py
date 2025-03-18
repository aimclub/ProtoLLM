import logging
from typing import Any, Iterable, List, Optional

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class RecursiveSplitter(RecursiveCharacterTextSplitter):
    """Splitting text by the given sequence of splitters."""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        kwargs["chunk_overlap"] = 0
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            ". ",
            ";",
            ", ",
            ".",
            ",",
            " ",
            "",
        ]
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        if self._length_function(text) < self._chunk_size:
            return [text]
        return self._split_text(text, self._separators)

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc = []
        for text in splits:
            merged_text = self._join_docs([*current_doc, text], separator)
            if (
                merged_text is None
                or self._length_function(merged_text) <= self._chunk_size
            ):
                current_doc.append(text)
                continue
            doc = self._join_docs(current_doc, separator)
            if doc is None and self._length_function(text) > self._chunk_size:
                logger.warning(
                    f"Created a chunk, which is longer than the specified {self._chunk_size}"
                )
            else:
                docs.append(doc)
            current_doc = [text]
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs
