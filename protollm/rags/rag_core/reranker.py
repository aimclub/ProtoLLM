import warnings
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate


class LLMReranker:
    def __init__(self, llm: LLM, prompt_template: PromptTemplate):
        """
        Reranker to change the order of documents using LLM.

        :param prompt_template: prompt template for reranking. It should contain the 'question' and 'context' fields
        """
        self._prompt_template = prompt_template
        # The retries number if the first model response was obtained in wrong format or have another anomalies
        # that are not characteristic for correct operation of the model in accordance with the prompt template
        self.num_retries = 3
        # The lower boundary of context LLM estimation
        self.qual_threshold = 2
        self._llm = llm

    def rerank_context(
        self, context: list[Document], user_query: str, top_k: int = 3
    ) -> list[Document]:
        ranking_prompts = [
            self._prompt_template.format(
                question=user_query,
                context=context_i.page_content
                + " Имя файла, откуда взят параграф "
                + context_i.metadata.get("source", "/None").split("/")[-1],
            )
            for context_i in context
        ]
        answers_ranking, bad_query = self._get_ranking_answer(ranking_prompts)
        if bad_query:
            fixed_answers = self._regenerate_answer(bad_query)
            answers_ranking += fixed_answers
        ext_context = self._extract_top_context(answers_ranking, top_k)
        if not ext_context:
            warnings.warn("Reranker does not support retrieved context")
        res_context = [context[ranking_prompts.index(i)] for i in ext_context]
        return res_context

    def _extract_top_context(
        self, pairs_to_rank: List[Tuple[str, int]], top_k: int
    ) -> list[str]:
        if not pairs_to_rank:
            return []
        pairs_to_rank.sort(key=lambda x: x[1], reverse=True)
        context = [x for x, y in pairs_to_rank if y >= self.qual_threshold]
        context = context[:top_k]
        return context

    def _get_ranking_answer(
        self, ranking_prompts: list[str]
    ) -> Tuple[list[Tuple[str, int]], list[str]]:
        answer = [self._llm.invoke(prompt) for prompt in ranking_prompts]
        answers_ranking = []
        bad_queries = []
        for i, ans_i in enumerate(answer):
            try:
                score = int(ans_i.split("ОЦЕНКА: ")[-1].strip())
                answers_ranking.append((ranking_prompts[i], score))
            except:
                bad_queries.append(ranking_prompts[i])
        return answers_ranking, bad_queries

    def _regenerate_answer(self, queries: list[str]) -> list[str]:
        fixed_queries = []
        for i in range(self.num_retries):
            good_res, bad_res = self._get_ranking_answer(queries)
            fixed_queries += good_res
            queries = bad_res
            if not bad_res:
                return fixed_queries
        return fixed_queries

    def merge_docs(
        self, query: str, contexts: list[list[Document]], top_k: int = 3
    ) -> list[Document]:
        ctx = []
        for context in zip(*contexts):
            ctx.extend(self.rerank_context(context, query, 1))

        if len(ctx) > top_k:
            return self.rerank_context(ctx, query, top_k)

        return ctx
