import zipfile
from pathlib import Path
from typing import Iterator, Union, Any, Optional, Sequence

from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document
from tqdm import tqdm

from protollm.raw_data_processing.docs_parsers.parsers import (
    ParsingScheme,
    DocType,
    BaseParser,
    PDFParser,
    WordDocumentParser,
)
from protollm.raw_data_processing.docs_parsers.utils.logger import ParsingLogger
from protollm.raw_data_processing.docs_parsers.utils.utilities import (
    correct_path_encoding,
)


class ZipLoader(BaseLoader):
    """
    Load files from zip archive into list of documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        byte_content: Optional[bytes] = None,
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
        """Initialize with a file path."""
        self.file_path = str(file_path)
        doc_type = BaseParser.get_doc_type(self.file_path)
        if doc_type is not DocType.zip:
            if doc_type is DocType.unsupported:
                raise ValueError("The file type is unsupported")
            else:
                raise ValueError(
                    f"The {doc_type} file type does not match the Loader! Use a suitable one."
                )
        self.byte_content = byte_content
        self._logger = parsing_logger or ParsingLogger(
            silent_errors=silent_errors, name=__name__
        )
        self.pdf_parser = PDFParser(
            pdf_parsing_scheme,
            pdf_extract_images,
            pdf_extract_tables,
            pdf_extract_formulas,
            pdf_remove_service_info,
        )
        self.word_doc_parser = WordDocumentParser(
            word_doc_parsing_scheme,
            word_doc_extract_images,
            word_doc_extract_tables,
            word_doc_extract_formulas,
            timeout_for_converting,
        )
        self._exclude_names = [Path(file).name for file in exclude_files]

    @property
    def logs(self):
        return self._logger.logs

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path"""
        content = self.byte_content or self.file_path
        with zipfile.ZipFile(content) as z:
            for info in tqdm(z.infolist(), desc="Zip processing", ncols=80):
                file_name = Path(correct_path_encoding(info.filename))
                if file_name.name in self._exclude_names:
                    continue
                path = str(Path(self.file_path, file_name))
                doc_type = BaseParser.get_doc_type(file_name)
                match doc_type:
                    case DocType.pdf:
                        _parser = self.pdf_parser
                    case DocType.docx | DocType.doc | DocType.odt | DocType.rtf:
                        _parser = self.word_doc_parser
                    case _:
                        if Path(file_name).suffix:
                            self._logger.info(
                                f"Skip file processing in zip, no suitable parser for {path}"
                            )
                        continue

                self._logger.info(f"Processing file in zip: {path}")
                blob = Blob.from_data(
                    z.open(info).read(), path=path, mime_type=doc_type.value
                )
                with self._logger.parsing_info_handler(path):
                    yield from _parser.lazy_parse(blob)
