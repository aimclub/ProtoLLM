from functools import partial
from pathlib import Path
from typing import Iterator, Union, Optional

from langchain_core.document_loaders import Blob
from langchain_core.documents import Document

from protollm.raw_data_processing.docs_parsers.utils.exceptions import EncodingError
from protollm.raw_data_processing.docs_parsers.parsers.base import BaseParser
from protollm.raw_data_processing.docs_parsers.parsers.entities import ParsingScheme
from protollm.raw_data_processing.docs_parsers.parsers.word_doc.utilities import (
    get_paragraphs,
    get_chapters,
    add_headings_hierarchy,
)
from protollm.raw_data_processing.docs_parsers.utils.utilities import (
    correct_path_encoding,
    is_bad_encoding,
)


class WordDocumentParser(BaseParser):
    """
    The parser provides a way to parse raw data from Word Document into one or more documents.
    """

    def __init__(
        self,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_formulas: bool = False,
        timeout_for_converting: Optional[int] = None,
    ):
        try:
            import protollm.raw_data_processing.docs_parsers.parsers.word_doc.docx_parsing
        except Exception as e:
            raise e
        # except ImportError as error:
        #     raise ImportError(
        #         f"{error.name} package not found, please try to install it with `pip install {error.name}`"
        #     )
        if parsing_scheme not in ParsingScheme.__members__:
            raise ValueError("Invalid parsing scheme")
        self.parsing_scheme = parsing_scheme
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_formulas = extract_formulas
        self.timeout = timeout_for_converting

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        from protollm.raw_data_processing.docs_parsers.parsers.word_doc.docx_parsing import (
            parse_docx_to_lines,
        )

        parse_docx_to_lines = partial(
            parse_docx_to_lines,
            extract_tables=self.extract_tables,
            extract_images=self.extract_images,
            extract_formulas=self.extract_formulas,
        )

        match blob.mimetype:
            case "doc" | "odt" | "rtf":
                from protollm.raw_data_processing.docs_parsers.parsers.converting import (
                    converted_file_to_docx,
                )

                with blob.as_bytes_io() as file_obj:
                    with converted_file_to_docx(
                        file_obj, timeout=self.timeout
                    ) as docx_file_obj:
                        lines, metadata = parse_docx_to_lines(docx_file_obj)
            case "docx":
                with blob.as_bytes_io() as docx_file_obj:
                    lines, metadata = parse_docx_to_lines(docx_file_obj)
            case _:
                raise ValueError("Invalid document type")

        if is_bad_encoding(lines):
            raise EncodingError(
                "It is impossible to parse the file due to uncertainty in the text encoding"
            )

        source = blob.source
        source = correct_path_encoding(source) if source is not None else ""
        file_name = Path(source).name

        if self.parsing_scheme == ParsingScheme.full:
            text = " ".join(lines)
            meta = {
                "page": "all",
                "headings": [],
                "source": source,
                "file_name": file_name,
            }
            yield Document(page_content=text, metadata=meta)
            return

        lines, metadata = add_headings_hierarchy(lines, metadata)

        match self.parsing_scheme:
            case ParsingScheme.lines:
                texts = lines
            case ParsingScheme.paragraphs:
                texts, metadata = get_paragraphs(lines, metadata)
            case ParsingScheme.chapters:
                texts, metadata = get_chapters(lines, metadata)
            case _:
                raise NotImplementedError(
                    f"{self.parsing_scheme} type of parsing scheme is not implemented"
                )

        for text, meta in zip(texts, metadata):
            yield Document(
                page_content=text,
                metadata={**meta, "source": source, "file_name": file_name},
            )
