from typing import Optional, Any

from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch.vectorstores import BaseRetrievalStrategy

from protollm.rags.stores.elasticsearch.settings import settings
from protollm.rags.stores.elasticsearch.retrieval_strategies import (
    BM25RetrievalStrategy,
)


def get_index_name(index: int) -> str:
    return f"index_v{index}"


def get_elasticsearch_store(
    index_name: str,
    es_url: str = settings.es_url,
    es_user: str = settings.es_user,
    es_password: str = settings.es_password,
    query_field: str = settings.content_field,
    strategy: BaseRetrievalStrategy = BM25RetrievalStrategy(),
    es_params: Optional[dict[str, Any]] = None,
) -> ElasticsearchStore:
    return ElasticsearchStore(
        index_name,
        es_url=es_url,
        es_user=es_user,
        es_password=es_password,
        query_field=query_field,
        strategy=strategy,
        es_params=es_params,
    )


def custom_query_for_metadata_mapping(query_body: dict, query: str) -> dict:
    """Custom query to be used in Elasticsearch with indexes that use langchain Document schema.
    This implies that all additional fields are stored in the metadata
    Args:
        query_body (dict): Elasticsearch query body.
        query (str): Query string.
    Returns:
        dict: Elasticsearch query body.
    """
    query = query_body["query"]
    if "multi_match" in query:
        if "fields" in query["multi_match"]:
            fields = []
            for field in query["multi_match"]["fields"]:
                if not (
                    field.startswith("metadata")
                    or field.startswith(settings.content_field)
                ):
                    field = "metadata." + field
                fields.append(field)

            query["multi_match"]["fields"] = fields

    return query_body
