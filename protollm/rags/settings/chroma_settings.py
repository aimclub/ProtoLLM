from os.path import dirname
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChromaSettings(BaseSettings):
    # Chroma DB settings
    chroma_host: str = "any"
    chroma_port: int = 8888
    allow_reset: bool = False

    # Documents collection's settings
    collection_name: str = "collection"
    collection_names_for_advance: list[str] = ["collection"]
    embedding_name: str = "intfloat/multilingual-e5-large"
    embedding_host: str = ""
    distance_fn: str = "cosine"

    # Documents' processing settings
    docs_processing_config: str = str(
        Path(
            dirname(dirname(__file__)), "/config_files/", "docs_processing_config.yaml"
        )
    )
    docs_collection_path: str = str(
        Path(dirname(dirname(dirname(__file__))), "/docs/", "example.docx")
    )

    model_config = SettingsConfigDict(
        env_file=Path(dirname(dirname(__file__)), "/config_files/", "chroma.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = ChromaSettings()
