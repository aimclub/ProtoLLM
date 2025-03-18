import logging
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Union

from langchain_community.vectorstores import utils
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from protollm.rags.pipeline.docs_processing.utils import get_loader

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class DocsExtractPipeline:
    def __init__(self, pipeline_settings: "PipelineSettings"):
        self._pipeline_settings = pipeline_settings

    def update_loader(self, **kwargs) -> "DocsExtractPipeline":
        self._pipeline_settings.update_loader_params(kwargs)
        return self

    def load_docs(
        self,
        docs_collection_path: Optional[Union[str, Path]] = None,
        byte_content: Optional[bytes] = None,
    ):
        logger.info("Initialize parsing process")
        if docs_collection_path is not None:
            self.update_loader(file_path=docs_collection_path)

        loader = get_loader(
            byte_content=byte_content, **self._pipeline_settings.loader_params
        )
        return loader.lazy_load()

    def go_to_next_step(
        self,
        docs_collection_path: Optional[Union[str, Path]] = None,
        byte_content: Optional[bytes] = None,
    ) -> "DocsTransformPipeline":
        return DocsTransformPipeline(
            self._pipeline_settings, self.load_docs(docs_collection_path, byte_content)
        )


class DocsTransformPipeline:
    def __init__(
        self, pipeline_settings: "PipelineSettings", docs_generator: Iterable[Document]
    ):
        self._pipeline_settings = pipeline_settings
        self._docs_generator = docs_generator

    def update_docs_transformers(self, **kwargs) -> "DocsTransformPipeline":
        self._pipeline_settings.update_transformer_params(kwargs)
        return self

    def transform(self, batch_size: int = 100) -> Iterable[Document]:
        logger.info("Initialize transformation process")
        transformers = self._pipeline_settings.transformers
        batch_size = max(batch_size, 1)
        while docs_batch := list(islice(self._docs_generator, batch_size)):
            for transformer in transformers:
                docs_batch = transformer.transform_documents(docs_batch)
            yield from docs_batch

    def go_to_next_step(self, batch_size: int = 100) -> "DocsLoadPipeline":
        return DocsLoadPipeline(self.transform(batch_size))


class DocsLoadPipeline:
    def __init__(self, docs_generator: Iterable[Document]):
        self._docs_generator = docs_generator

    def load(self, store: VectorStore, loading_batch_size: int = 32) -> None:
        logger.info("Initialize loading process")

        # Add processed documents to the store
        loading_batch_size = max(loading_batch_size, 1)
        while docs_batch := list(islice(self._docs_generator, loading_batch_size)):
            # TODO: replace filtering. It loses important metadatas. Use DocsTransformer to transform metadata
            store.add_documents(utils.filter_complex_metadata(docs_batch))
