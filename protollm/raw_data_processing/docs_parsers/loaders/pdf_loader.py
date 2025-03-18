from pathlib import Path
from typing import Iterator, Union, Any, Optional

from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document

from protollm.raw_data_processing.docs_parsers.parsers import (
    PDFParser,
    ParsingScheme,
    DocType,
)
from protollm.raw_data_processing.docs_parsers.utils.logger import ParsingLogger


class PDFLoader(BaseLoader):
    """
    Load PDF into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_formulas: bool = False,
        remove_headers: bool = False,
        parsing_logger: Optional[ParsingLogger] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = PDFParser.get_doc_type(self.file_path)
        if doc_type is not DocType.pdf:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._logger = parsing_logger or ParsingLogger(name=__name__)
        self.parser = PDFParser(
            parsing_scheme,
            extract_images,
            extract_tables,
            extract_formulas,
            remove_headers,
        )

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        if self.byte_content is None:
            blob = Blob.from_path(self.file_path)
        else:
            blob = Blob.from_data(
                self.byte_content, path=self.file_path, mime_type=DocType.pdf.value
            )
        with self._logger.parsing_info_handler(self.file_path):
            yield from self.parser.lazy_parse(blob)
