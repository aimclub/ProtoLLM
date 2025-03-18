from pathlib import Path
from typing import Iterator, Union, Any, Optional

from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document

from protollm.raw_data_processing.docs_parsers.parsers import (
    WordDocumentParser,
    ParsingScheme,
    DocType,
)
from protollm.raw_data_processing.docs_parsers.utils.logger import ParsingLogger


def preprocess_documents(func):
    def wrapper(self, *args, **kwargs):
        documents = func(self, *args, **kwargs)
        for doc in documents:
            if len(doc.page_content) < 15:
                continue
            if sum(c.isdigit() for c in doc.page_content) / len(doc.page_content) > 0.2:
                continue
            yield doc

    return wrapper


class WordDocumentLoader(BaseLoader):
    """
    Load Word Document into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
        parsing_logger: Optional[ParsingLogger] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = WordDocumentParser.get_doc_type(self.file_path)
        if doc_type not in [DocType.docx, DocType.doc, DocType.odt, DocType.rtf]:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._doc_type = doc_type.value
        self._logger = parsing_logger or ParsingLogger(name=__name__)
        self.parser = WordDocumentParser(
            parsing_scheme,
            extract_images,
            extract_tables,
            extract_formulas,
            timeout_for_converting,
        )

    @property
    def logs(self):
        return self._logger.logs

    @preprocess_documents
    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        if self.byte_content is None:
            blob = Blob.from_path(self.file_path, mime_type=self._doc_type)
        else:
            blob = Blob.from_data(
                self.byte_content, path=self.file_path, mime_type=self._doc_type
            )
        with self._logger.parsing_info_handler(self.file_path):
            yield from self.parser.lazy_parse(blob)
