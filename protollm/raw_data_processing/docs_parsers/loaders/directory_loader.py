from pathlib import Path
from typing import Iterator, Union, Any, Optional, Sequence

from langchain_community.document_loaders.directory import _is_visible
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from tqdm import tqdm

from protollm.raw_data_processing.docs_parsers.parsers import (
    ParsingScheme,
    DocType,
    BaseParser,
)
from protollm.raw_data_processing.docs_parsers.utils.logger import ParsingLogger
from protollm.raw_data_processing.docs_parsers.loaders.pdf_loader import PDFLoader
from protollm.raw_data_processing.docs_parsers.utils.utilities import (
    correct_path_encoding,
)
from protollm.raw_data_processing.docs_parsers.loaders.doc_loader import (
    WordDocumentLoader,
)
from protollm.raw_data_processing.docs_parsers.loaders.zip_loader import ZipLoader


class RecursiveDirectoryLoader(BaseLoader):
    """
    Load files from directory into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        pdf_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        pdf_extract_images: bool = False,
        pdf_extract_tables: bool = False,
        pdf_extract_formulas: bool = False,
        pdf_remove_service_info: bool = False,
        word_doc_parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        word_doc_extract_images: bool = False,
        word_doc_extract_tables: bool = False,
        word_doc_extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
        exclude_files: Sequence[Union[Path, str]] = (),
        parsing_logger: Optional[ParsingLogger] = None,
        silent_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with a directory path."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.file_path}'")
        if not self.file_path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.file_path}'")

        self._logger = parsing_logger or ParsingLogger(
            silent_errors=silent_errors, name=__name__
        )

        self._pdf_kwargs = {
            "parsing_scheme": pdf_parsing_scheme,
            "extract_images": pdf_extract_images,
            "extract_tables": pdf_extract_tables,
            "extract_formulas": pdf_extract_formulas,
            "remove_headers": pdf_remove_service_info,
            "parsing_logger": self._logger,
        }
        self._word_doc_kwargs = {
            "parsing_scheme": word_doc_parsing_scheme,
            "extract_images": word_doc_extract_images,
            "extract_tables": word_doc_extract_tables,
            "extract_formulas": word_doc_extract_formulas,
            "timeout_for_converting": timeout_for_converting,
            "parsing_logger": self._logger,
        }
        self._zip_kwargs = {
            "pdf_parsing_scheme": pdf_parsing_scheme,
            "pdf_extract_images": pdf_extract_images,
            "pdf_extract_tables": pdf_extract_tables,
            "pdf_extract_formulas": pdf_extract_formulas,
            "pdf_remove_service_info": pdf_remove_service_info,
            "word_doc_parsing_scheme": word_doc_parsing_scheme,
            "word_doc_extract_images": word_doc_extract_images,
            "word_doc_extract_tables": word_doc_extract_tables,
            "word_doc_extract_formulas": word_doc_extract_formulas,
            "timeout_for_converting": timeout_for_converting,
            "exclude_files": exclude_files,
            "parsing_logger": self._logger,
        }
        self._exclude_names = [Path(file).name for file in exclude_files]

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        paths = [
            path
            for path in self.file_path.rglob("**/[!.]*")
            if path.is_file()
            and _is_visible(path)
            and path.name not in self._exclude_names
            and correct_path_encoding(path.name) not in self._exclude_names
        ]
        for path in tqdm(paths, desc="Directory processing", ncols=80):
            doc_type = BaseParser.get_doc_type(path)
            match doc_type:
                case DocType.pdf:
                    _loader = PDFLoader(path, **self._pdf_kwargs)
                case DocType.docx | DocType.doc | DocType.odt | DocType.rtf:
                    _loader = WordDocumentLoader(path, **self._word_doc_kwargs)
                case DocType.zip:
                    _loader = ZipLoader(path, **self._zip_kwargs)
                case _:
                    self._logger.info(
                        f"Skip file processing, no suitable loader for {path}"
                    )
                    continue

            self._logger.info(f"Processing file: {path}")
            yield from _loader.lazy_load()
