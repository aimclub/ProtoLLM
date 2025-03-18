import html
import re
from typing import Optional, Iterator
from uuid import uuid4

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.table import Table, _Cell, _Row
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from ftfy import fix_text
from tabulate import tabulate

from protollm.raw_data_processing.docs_parsers.parsers.utilities import is_bulleted_text
from protollm.raw_data_processing.docs_parsers.parsers.word_doc.docx_parsing_config import (
    DocxParsingConfig,
)
from protollm.raw_data_processing.docs_parsers.parsers.word_doc.xml import (
    process_paragraph_body,
)


def _get_list_level(split_text: list[str], level: int = -1) -> int:
    if not re.search(r"[а-яА-ЯёЁ]", "".join(split_text[1:])):
        return level
    if not split_text:
        return level
    if not split_text[0].isdigit():
        return level
    else:
        level += 1
        return _get_list_level(split_text[1:], level)


def _get_urls(paragraph: Paragraph) -> dict[str, str]:
    urls = {}
    for item in paragraph.iter_inner_content():
        if isinstance(item, Hyperlink):
            text = item.text
            url = item.url
            if not text or not url:
                continue
            urls[text] = url
    return urls


def _get_metadata(paragraph: Optional[Paragraph] = None) -> dict:
    bold = False
    font_size = -1
    urls = {}
    list_level = _get_list_level([])
    is_bullet_list = False
    is_centered = False
    first_line_indent = 0

    if paragraph is not None:
        bold = paragraph.runs[0].bold or bold if paragraph.runs else bold
        bold = paragraph.style.font.bold or bold

        font_size = (
            paragraph.runs[0].font.size or font_size if paragraph.runs else font_size
        )

        urls.update(_get_urls(paragraph))

        split_text = re.split("[ .\xa0]", paragraph.text)
        list_level = _get_list_level(split_text)

        is_bullet_list = is_bulleted_text(paragraph.text) | (
            "<w:numPr>" in paragraph._element.xml
        )

        is_centered = paragraph.paragraph_format.alignment is WD_ALIGN_PARAGRAPH.CENTER

        paragraph_first_line_indent = paragraph.paragraph_format.first_line_indent
        first_line_indent = (
            paragraph_first_line_indent.emu
            if paragraph_first_line_indent is not None
            else first_line_indent
        )

    return {
        "bold": bold,
        "list_level": list_level,
        "is_bullet_list": is_bullet_list,
        "urls": urls,
        "is_centered": is_centered,
        "first_line_indent": first_line_indent,
        "font_size": font_size,
        "images": {},
        "formulas": {},
        "tables": {},
    }


def _process_paragraph(
    paragraph: Paragraph, parsing_config: DocxParsingConfig
) -> tuple[str, dict]:
    paragraph_text, paragraph_metadata = process_paragraph_body(
        paragraph, parsing_config=parsing_config
    )
    paragraph_text = fix_text(paragraph_text).replace("\xa0", " ")
    paragraph_text = " ".join(paragraph_text.split())

    metadata = _get_metadata(paragraph)
    metadata.update(paragraph_metadata)

    return paragraph_text, metadata


def _convert_to_html(
    table: Table, parsing_config: DocxParsingConfig, is_nested: bool = False
) -> tuple[str, list[dict]]:
    cells_metadata = []

    def iter_cell_block_items(cell: _Cell) -> Iterator[str]:
        for block_item in cell.iter_inner_content():
            if isinstance(block_item, Table):
                inner_html_table, inner_cells_metadata = _convert_to_html(
                    block_item, parsing_config, is_nested=True
                )
                cells_metadata.extend(inner_cells_metadata)
                yield inner_html_table
            elif isinstance(block_item, Paragraph):
                paragraph_text, metadata = _process_paragraph(
                    block_item, parsing_config
                )
                cells_metadata.append(metadata)
                yield f"{html.escape(paragraph_text)}"

    def iter_cells(row: _Row) -> Iterator[str]:
        return ("\n".join(iter_cell_block_items(cell)) for cell in row.cells)

    return (
        tabulate(
            [list(iter_cells(row)) for row in table.rows],
            headers=[] if is_nested else "firstrow",
            tablefmt="unsafehtml",
        ),
        cells_metadata,
    )


def _process_table(table: Table, parsing_config: DocxParsingConfig) -> tuple[str, dict]:
    html_table, cells_metadata = _convert_to_html(table, parsing_config)
    table_name = "table-" + str(uuid4())
    metadata = _get_metadata()
    metadata["tables"] = {table_name: html_table}
    for cell_meta in cells_metadata:
        for key in ["images", "formulas", "urls"]:
            metadata[key] = {**metadata.get(key, {}), **cell_meta.get(key, {})}

    return html_table, metadata  # TODO: f'{{{table_name}}}' instead of html_table


def parse_docx_to_lines(
    stream,
    extract_tables: bool = False,
    extract_images: bool = False,
    extract_formulas: bool = False,
) -> tuple[list[str], list[dict]]:
    lines, metadata = [], []
    document = docx.Document(stream)
    parsing_config = DocxParsingConfig(document, extract_images, extract_formulas)
    for section in document.sections:
        for block_item in section.iter_inner_content():
            if isinstance(block_item, Paragraph):
                line, meta = _process_paragraph(block_item, parsing_config)
            elif isinstance(block_item, Table) and extract_tables:
                line, meta = _process_table(block_item, parsing_config)
            else:
                continue

            if not line:
                continue

            lines.append(line)
            metadata.append(meta)

    return lines, metadata
