import ast

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate


class Planner:

    def __init__(self, llm: LLM, prompt_template: PromptTemplate) -> None:
        self._prompt_template = prompt_template
        self._llm = llm

    def generate_answer(self, query: str | list[str]) -> list:
        if isinstance(query, str):
            query = [query]
        inst_query = [self._prompt_template.format(context=i) for i in query]
        answer = [self._llm.invoke(prompt) for prompt in inst_query]
        good_answers, bad_answers = self._extract_planner_queries(answer)
        if len(bad_answers) > 0:
            updated_answers = self._regenerate_answer(bad_answers)
            good_answers += updated_answers
        return good_answers

    def _extract_planner_queries(self, answer: list):
        bad_idx = []
        for i, ans in enumerate(answer):
            try:
                result = ast.literal_eval(
                    ans.content.split("ЗАПРОСЫ:")[1].strip().split("]")[0] + "]"
                )
                answer[i] = result
            except:
                bad_idx.append(i)
        return [answer[i] for i in range(len(answer)) if i not in bad_idx], [
            answer[i] for i in range(len(answer)) if i in bad_idx
        ]

    def _regenerate_answer(self, query: list, retries: int = 3):
        fixed_queries = []
        for trial in range(retries):
            result = [self._llm.invoke(prompt) for prompt in query]
            good_res, bad_res = self._extract_planner_queries(result)
            fixed_queries += good_res
            query = bad_res
            if len(bad_res) == 0:
                return fixed_queries
        return fixed_queries
