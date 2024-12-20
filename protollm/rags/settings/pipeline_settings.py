import warnings
from copy import deepcopy
from typing import TextIO, Optional, Any, Type

import inspect

import yaml
from langchain_core.documents import BaseDocumentTransformer
from langchain_text_splitters import TextSplitter
from transformers import AutoTokenizer, logging

from protollm.rags.pipeline.docs_processing.entities import transformer_object_dict
from protollm.rags.pipeline.docs_processing.exceptions import TransformerNameError
from protollm.rags.pipeline.docs_processing.models import ConfigFile


def _get_params_for_transformer(params: dict[str, Any],
                                transformer_class: Type[BaseDocumentTransformer]) -> dict[str, Any]:
    text_splitter_params = inspect.signature(TextSplitter.__init__).parameters.keys()
    transformer_params = {key: value for key, value in params.items()
                          if key in inspect.signature(transformer_class.__init__).parameters.keys()
                          or key in text_splitter_params}
    return transformer_params


class PipelineSettings:
    def __init__(self, config: Optional[ConfigFile] = None):
        self.config = config
        self.loader_params = config
        self.transformers = config

    @classmethod
    def config_from_file(cls, config_file: str):
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config_dict = ConfigFile.model_validate(yaml_config)
        for splitter in config_dict.splitter:
            if splitter.splitter_name not in transformer_object_dict:
                raise TransformerNameError(f'There is no DocumentTransformer '
                                           f'related to the name: {splitter.splitter_name}')
        return cls(config=config_dict)

    def update_transformer_params(self, new_params: dict[str, Any]):
        for transformer_class, transformer_params in self._transformers:
            new_param_values = _get_params_for_transformer(new_params, transformer_class)
            transformer_params.update(new_param_values)

    def update_loader_params(self, new_params: dict[str, Any]):
        self._loader_params.update(new_params)

    @property
    def loader_params(self) -> dict[str, Any]:
        return self._loader_params

    @loader_params.setter
    def loader_params(self, config: Optional[ConfigFile]):
        loader_params = deepcopy(config.loader.parsing_params)
        loader_params['file_path'] = config.loader.file_path
        self._loader_params = loader_params

    @property
    def config(self) -> ConfigFile:
        return self._config

    @config.setter
    def config(self, config: Optional[ConfigFile]):
        self._config = deepcopy(config)

    @property
    def transformers(self) -> list[BaseDocumentTransformer]:
        docs_transformers = []
        tokenizer = None
        if self.config.tokenizer is not None:
            logging.set_verbosity_error()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        for transformer_class, transformer_params in self._transformers:
            if tokenizer is None or not hasattr(transformer_class, 'from_huggingface_tokenizer'):
                transformer = transformer_class(**transformer_params)
            else:
                transformer = transformer_class.from_huggingface_tokenizer(
                    tokenizer,
                    **transformer_params
                )
            docs_transformers.append(transformer)
        return docs_transformers

    @transformers.setter
    def transformers(self, config: Optional[ConfigFile]):
        self._transformers = []
        if config is None:
            return
        self._transformers.append((transformer_object_dict['recursive_character'], {}))
        for splitter in config.splitter:
            splitter_params = splitter.splitter_params
            transformer_class = transformer_object_dict[splitter.splitter_name]
            transformer_params = _get_params_for_transformer(splitter_params, transformer_class)
            if splitter.splitter_name != 'recursive_character':
                self._transformers.append((transformer_class, transformer_params))
            else:
                self._transformers[0] = (transformer_class, transformer_params)
