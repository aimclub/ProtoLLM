from typing import Any, List

from pydantic import BaseModel


class ConfigLoader(BaseModel):
    file_path: str = ""
    save_path: str = ""
    loader_name: str
    parsing_params: dict[str, Any] = dict()


class ConfigSplitter(BaseModel):
    splitter_name: str | None = None
    splitter_params: dict[str, Any] = dict()


class ConfigFile(BaseModel):
    loader: ConfigLoader
    splitter: List[ConfigSplitter] = []
    tokenizer: str | None = None
