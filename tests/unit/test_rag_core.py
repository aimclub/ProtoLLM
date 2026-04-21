from typing import Any, cast


class DummyLLM:
    def __init__(self, response: str):
        self._response = response

    def invoke(self, prompt: str) -> str:
        return self._response


class DummyRetriever:
    def __init__(self, docs: list[Any]):
        self._docs = docs

    # Align with DocRetriever API used by RetrievingPipeline
    def retrieve_top(self, collection_name: str, query: str, filter=None):
        return list(self._docs)


class DummyDoc:
    def __init__(self, text: str):
        self.page_content = text


def test_run_rag_basic_flow(monkeypatch):
    # Arrange
    from protollm.rags.rag_core import utils as rag_utils

    docs = [DummyDoc("A"), DummyDoc("B")]
    retrievers = [DummyRetriever(docs)]
    llm = DummyLLM("ok")

    # Act
    result = rag_utils.run_rag(
        user_prompt="q",
        llm=cast(Any, llm),
        retrievers=cast(Any, retrievers),
        collection_names=["col"],
        do_reranking=False,
    )

    # Assert
    assert result == "ok"


def test_run_multiple_rag_merges_contexts(monkeypatch):
    # Arrange
    from protollm.rags.rag_core import utils as rag_utils

    class DummyPipeline:
        def __init__(self, docs):
            self._docs = docs

        def get_retrieved_docs(self, user_prompt: str):
            return list(self._docs)

    # force reranker.merge_docs to interleave/return concatenated docs
    class DummyReranker:
        def __init__(self, *args, **kwargs):
            pass

        def merge_docs(self, user_prompt, contexts):
            merged = []
            for row in zip(*contexts):
                merged.extend(list(row))
            return merged

    monkeypatch.setitem(rag_utils.__dict__, "LLMReranker", DummyReranker)

    pipelines = [
        DummyPipeline([DummyDoc("A1"), DummyDoc("A2")]),
        DummyPipeline([DummyDoc("B1"), DummyDoc("B2")]),
    ]

    # Act
    res = rag_utils.run_multiple_rag(
        user_prompt="q",
        llm=cast(Any, DummyLLM("ok")),
        retriever_pipelines=cast(Any, pipelines),
    )

    # Assert
    assert res == "ok"


def test_run_multiple_rag_handles_different_context_lengths(monkeypatch):
    # Arrange
    from protollm.rags.rag_core import utils as rag_utils

    class DummyPipeline:
        def __init__(self, docs):
            self._docs = docs

        def get_retrieved_docs(self, user_prompt: str):
            return list(self._docs)

    class DummyReranker:
        def __init__(self, *args, **kwargs):
            pass

        def merge_docs(self, user_prompt, contexts):
            # just return flattened
            out = []
            for ctx in contexts:
                out.extend(ctx)
            return out

    monkeypatch.setitem(rag_utils.__dict__, "LLMReranker", DummyReranker)

    pipelines = [
        DummyPipeline([DummyDoc("A1")]),
        DummyPipeline([DummyDoc("B1"), DummyDoc("B2"), DummyDoc("B3")]),
    ]

    # Act
    res = rag_utils.run_multiple_rag(
        user_prompt="q",
        llm=cast(Any, DummyLLM("ok")),
        retriever_pipelines=cast(Any, pipelines),
    )

    # Assert
    assert res == "ok"
