import re
from typing import Any, Iterable, Optional

from langchain_text_splitters import TextSplitter

from protollm.raw_data_processing.docs_transformers.utilities import (
    fix_list_dots_separators,
)


class SentencesSplitter(TextSplitter):
    def __init__(self, separators: Optional[Iterable[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._separators = separators or [r"\. (?<![0-9]\. | ^[0-9].)"]

    def split_text(self, text: str) -> list[str]:
        sentences_lst = [
            x.strip() for x in re.split(" | ".join(self._separators), text)
        ]
        return fix_list_dots_separators(sentences_lst)
