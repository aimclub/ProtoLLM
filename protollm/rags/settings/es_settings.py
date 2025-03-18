import json
from os.path import dirname
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_PATH = Path(dirname(dirname(__file__)), "/stores/elasticsearch/configs")


class ElasticsearchSettings(BaseSettings):
    es_host: str = "elasticDUMB"
    es_port: int = 9201

    es_user: str = "elastic"
    es_password: str = "admin"

    @computed_field
    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    es_index_mappings: dict = json.loads(
        Path(CONFIG_PATH, "index_mappings.json").read_text(encoding="utf-8")
    )
    es_index_settings: dict = json.loads(
        Path(CONFIG_PATH, "index_settings.json").read_text(encoding="utf-8")
    )
    es_query_template: dict = json.loads(
        Path(CONFIG_PATH, "query_template.json").read_text(encoding="utf-8")
    )
    es_query_all_hits: dict = json.loads(
        Path(CONFIG_PATH, "query_all_hits.json").read_text(encoding="utf-8")
    )

    metadata_fields: list[str] = list(
        es_index_mappings["properties"]["metadata"]["properties"].keys()
    )
    content_field: str = "paragraph"

    model_config = SettingsConfigDict(
        env_file=Path(Path(__file__).parent.parent, "/configs/elastic.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = ElasticsearchSettings()
