from typing import Any, Optional, List
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.language_models import LanguageModelInput


class Dataset:
    """
    A class to load and manipulate datasets.
    """

    def __init__(
        self,
        path: str,
        labels_col: bool = False,
        data_col: str = "content",
        max_chunk_size: int = 200000,
    ):
        self.path = path
        self.labels_col = labels_col
        self.data_col = data_col
        self.max_chunk_size = max_chunk_size
        self.data = None
        self.labeled_data = None
        self.load()

    def load(self):
        # check if file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File {self.path} does not exist")

        # load data
        if self.path.endswith(".json"):
            self.data = pd.read_json(self.path)
        elif self.path.endswith(".csv"):
            self.data = pd.read_csv(self.path)
        else:
            raise ValueError(f"Unsupported file type: {self.path}")

        # Verify data_col exists in the DataFrame
        if self.data_col not in self.data.columns:
            available_cols = list(self.data.columns)
            raise ValueError(
                f"Column '{self.data_col}' not found in dataset. Available columns: {available_cols}"
            )

        if self.labels_col:
            self.get_labeled_data()

        # Checking that some elements in data_col are larger than max_chunk_size
        if self.data[self.data_col].apply(lambda x: len(x) > self.max_chunk_size).any():
            self.chunk_context(self.data_col, self.max_chunk_size)

    def get_labeled_data(self):
        self.labeled_data = self.data[self.data[self.labels_col].notna()]
        self.data = self.data[self.data[self.labels_col].isna()]

    def train_test_split(self, test_size: float = 0.2):
        raise NotImplementedError("Not implemented")

    def chunk_context(self, column_name: str, max_chunk_size: int):
        """
        Splits the text in the specified column into chunks if the text size exceeds max_chunk_size.

        :param column_name: The name of the column to process.
        :param max_chunk_size: The maximum size of each chunk.
        """
        if column_name not in self.data.columns:
            raise ValueError(f"Column {column_name} does not exist in the dataset")

        def split_into_chunks(text, max_size):
            # Split text into chunks of max_size
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        self.data[column_name] = self.data[column_name].apply(
            lambda x: (
                split_into_chunks(x, max_chunk_size)
                if isinstance(x, str) and len(x) > max_chunk_size
                else x
            )
        )


class VLLMChatOpenAI(ChatOpenAI):
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        # Call the parent class method to get the initial payload
        request_payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Check for deprecated 'max_completion_tokens' and replace it with 'max_tokens'
        if "max_completion_tokens" in request_payload:
            request_payload["max_tokens"] = request_payload.pop("max_completion_tokens")

        return request_payload
