from contextlib import contextmanager
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import BinaryIO, Generator, Union, Optional

from protollm.raw_data_processing.docs_parsers.parsers.converting.converting import (
    _convert_with_soffice,
)
from protollm.raw_data_processing.docs_parsers.parsers.entities import ConvertingDocType


@contextmanager
def converted_file(
    stream,
    target_doc_type: Union[str, ConvertingDocType] = ConvertingDocType.docx,
    timeout: Optional[int] = None,
) -> Generator[BinaryIO, None, None]:
    if target_doc_type not in ConvertingDocType.__members__:
        raise ValueError("Invalid target document type")
    target_doc_type = f"{target_doc_type}"

    with TemporaryDirectory() as tmp_dir:
        tmp_file = NamedTemporaryFile(delete=False, dir=tmp_dir)
        tmp_file.write(stream.read())
        tmp_file.close()
        tmp_file_path = tmp_file.name

        _convert_with_soffice(
            filename=tmp_file_path,
            output_directory=tmp_dir,
            target_doc_type=target_doc_type,
            timeout=timeout,
        )
        converted_file_path = ".".join((tmp_file_path, target_doc_type))

        with open(converted_file_path, "rb") as f:
            yield f
