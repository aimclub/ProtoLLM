from os.path import dirname
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class ChromaSettings(BaseSettings):
    # Chroma DB settings
    chroma_host: str = '127.0.0.1'
    chroma_port: int = 8888
    allow_reset: bool = False

    # Documents collection's settings
    collection_name: str = 'collection'
    collection_names_for_advance: list[str] = ['collection']
    embedding_name: str = 'intfloat/multilingual-e5-large'
    embedding_host: str = ''
    distance_fn: str = 'cosine'

    # Documents' processing settings
    docs_processing_config: Optional[str] = None
    docs_collection_path: str = str(Path(dirname(dirname(__file__))) / 'docs' / 'example.docx')


settings = ChromaSettings()
