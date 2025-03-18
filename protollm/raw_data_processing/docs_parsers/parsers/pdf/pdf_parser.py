import re
import warnings
from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import Blob
from langchain_core.documents import Document

from protollm.raw_data_processing.docs_parsers.utils.exceptions import (
    EncodingError,
    ChaptersExtractingFailedWarning,
)
from protollm.raw_data_processing.docs_parsers.parsers.base import BaseParser
from protollm.raw_data_processing.docs_parsers.parsers.entities import ParsingScheme
from protollm.raw_data_processing.docs_parsers.parsers.utilities import (
    CONTENTS_KEYWORDS,
)
from protollm.raw_data_processing.docs_parsers.utils.utilities import (
    correct_path_encoding,
    is_bad_encoding,
)


class PDFParser(BaseParser):
    """
    The parser provides a way to parse raw data from PDF into one or more documents.
    """

    def __init__(
        self,
        parsing_scheme: Union[ParsingScheme, str] = ParsingScheme.lines,
        extract_images: bool = False,
        extract_tables: bool = False,
        parse_formulas: bool = False,
        remove_service_info: bool = False,
    ):
        try:
            import protollm.raw_data_processing.docs_parsers.parsers.pdf.utilities
        except ImportError as error:
            raise ImportError(
                f"{error.name} package not found, please try to install it with `pip install {error.name}`"
            )
        if parsing_scheme not in ParsingScheme.__members__:
            raise ValueError("Invalid parsing scheme")
        self.parsing_scheme = parsing_scheme
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.parse_formulas = parse_formulas
        self.remove_service_info = remove_service_info

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        from protollm.raw_data_processing.docs_parsers.parsers.pdf.utilities import (
            extract_by_lines,
        )

        source = blob.source
        source = correct_path_encoding(source) if source is not None else ""
        file_name = Path(source).name

        with blob.as_bytes_io() as pdf_file_obj:
            lines, metadata = extract_by_lines(
                pdf_file_obj,
                parse_images=self.extract_images,
                parse_tables=self.extract_tables,
                parse_formulas=self.parse_formulas,
                remove_service_info=self.remove_service_info,
            )

        if not lines:
            return

        if is_bad_encoding(lines):
            raise EncodingError(
                "It is impossible to parse the file due to uncertainty in the text encoding"
            )

        # Get info about maximum titles hierarchy level
        max_hierarchy_lvl = max([len(x["headings"]) for x in metadata])

        is_heading_extracting_correct = True
        for meta in metadata:
            if not meta["is_heading_extracting_correct"]:
                is_heading_extracting_correct = False
                break

        if max_hierarchy_lvl != 0:
            upd_lines = []
            upd_metadata = []
            for text, meta in zip(lines, metadata):
                if self.remove_service_info:
                    if (
                        len(meta["headings"]) > 0
                        and meta["headings"][0].lower() in CONTENTS_KEYWORDS
                    ):  # len(meta['headings']) == 0 or
                        continue
                upd_lines.append(text)
                upd_metadata.append(meta)
            lines = upd_lines
            metadata = upd_metadata
        else:
            for i in range(len(metadata)):
                metadata[i]["headings"].append("Документ")

        match self.parsing_scheme:
            case ParsingScheme.lines:
                for text, meta in zip(lines, metadata):
                    yield Document(
                        page_content=text,
                        metadata={**meta, "source": source, "file_name": file_name},
                    )
            case ParsingScheme.full:
                text = " ".join([x.strip() for x in lines])
                pattern = r"(?<=[А-Яа-яёЁ])-\s"
                text = re.sub(pattern, "", text)
                meta = {
                    "page": "all",
                    "source": source,
                    "file_name": file_name,
                    "is_heading_extracting_correct": is_heading_extracting_correct,
                }
                yield Document(page_content=text, metadata=meta)
            case ParsingScheme.chapters:
                if not is_heading_extracting_correct:
                    warnings.warn(
                        "Can not correctly extract chapters due to complex document structure",
                        category=ChaptersExtractingFailedWarning,
                    )
                text_lst = []
                heading = ""
                for text, meta in zip(lines, metadata):
                    if len(meta["headings"]) != 0:  # text is a part of some chapter
                        if (
                            meta["headings"][0] == heading
                        ):  # it has the same chapter's heading
                            text_lst.append(
                                text
                            )  # add element's text to the document's content
                        elif heading != "":  # other heading
                            document_text = " ".join(text_lst)
                            document_meta = {
                                "heading": heading,
                                "source": source,
                                "file_name": file_name,
                                "is_heading_extracting_correct": is_heading_extracting_correct,
                            }

                            text_lst = [text]
                            heading = meta["headings"][0]
                            pattern = r"(?<=[А-Яа-яёЁ])-\s"
                            document_text = re.sub(pattern, "", document_text)
                            yield Document(
                                page_content=document_text, metadata=document_meta
                            )
                        else:
                            text_lst = [text]
                            heading = meta["headings"][0]
                if len(text_lst) != 0:
                    document_text = " ".join(text_lst)
                    pattern = r"(?<=[А-Яа-яёЁ])-\s"
                    document_text = re.sub(pattern, "", document_text)
                    document_meta = {
                        "heading": heading,
                        "source": source,
                        "file_name": file_name,
                        "is_heading_extracting_correct": is_heading_extracting_correct,
                    }

                    yield Document(page_content=document_text, metadata=document_meta)
            case ParsingScheme.paragraphs:
                text_lst = []
                paragraph = -1
                document_meta = {
                    "source": source,
                    "file_name": file_name,
                    "is_heading_extracting_correct": is_heading_extracting_correct,
                }
                for text, meta in zip(lines, metadata):
                    if len(meta["headings"]) != 0:  # text is a part of some chapter
                        if (
                            meta["paragraph"] != -1
                        ):  # text is a part of some paragraph (not header)
                            if (
                                meta["paragraph"] == paragraph
                            ):  # it has the same paragraph's number
                                text_lst.append(
                                    text
                                )  # add element's text to the document's content
                                document_meta = {
                                    **meta,
                                    "source": source,
                                    "file_name": file_name,
                                    "is_heading_extracting_correct": is_heading_extracting_correct,
                                }
                            elif paragraph != -1:  # other paragraph
                                document_text = " ".join(text_lst)
                                text_lst = [text]
                                paragraph = meta["paragraph"]
                                pattern = r"(?<=[А-Яа-яёЁ])-\s"
                                document_text = re.sub(pattern, "", document_text)
                                yield Document(
                                    page_content=document_text, metadata=document_meta
                                )
                            else:
                                text_lst = [text]
                                paragraph = meta["paragraph"]
                if len(text_lst) != 0:
                    document_text = " ".join(text_lst)
                    pattern = r"(?<=[А-Яа-яёЁ])-\s"
                    document_text = re.sub(pattern, "", document_text)
                    yield Document(page_content=document_text, metadata=document_meta)
            case _:
                raise NotImplementedError(
                    f"{self.parsing_scheme} type of parsing scheme is not implemented"
                )
