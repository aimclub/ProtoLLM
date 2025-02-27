import os
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / 'configs'


class ElasticsearchSettings:
    es_host: str = os.environ.get("ELASTIC_HOST", "")
    es_port: int = os.environ.get("ELASTIC_PORT", 80)
    es_url: str = f"http://{es_host}:{es_port}"
    es_user: str = os.environ.get("ELASTIC_USER", "")
    es_password: str = os.environ.get("ELASTIC_PASSWORD", "")

    es_index_mappings: dict = json.loads(Path(CONFIG_PATH, 'index_mappings.json').read_text(encoding="utf-8"))
    es_index_settings: dict = json.loads(Path(CONFIG_PATH, 'index_settings.json').read_text(encoding="utf-8"))
    es_query_template: dict = json.loads(Path(CONFIG_PATH, 'query_template.json').read_text(encoding="utf-8"))
    es_query_all_hits: dict = json.loads(Path(CONFIG_PATH, 'query_all_hits.json').read_text(encoding="utf-8"))

    metadata_fields: list[str] = list(es_index_mappings['properties']['metadata']['properties'].keys())
    content_field: str = 'paragraph'


settings = ElasticsearchSettings()
