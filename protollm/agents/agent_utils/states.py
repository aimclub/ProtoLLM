import operator
from pathlib import Path
from typing import Annotated, List, Set, Tuple

from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]

    past_steps: Annotated[Set[Tuple[str, str]], operator.or_]
    nodes_calls: Annotated[Set[Tuple[str, tuple]], operator.or_]

    next: str
    response: str
    visualization: str
    language: str
    translation: str
    automl_results: str
    last_memory: str
    metadata: Annotated[dict, operator.or_]

    parallel_tasks: dict
    attached_img: Path


def load_summary(user_id: str) -> str:
    from protollm.agents.universal_agents import store

    namespace = (user_id, "memory")
    item = store.get(namespace, "latest-summary")
    return item.value.get("summary", "") if item else ""


def initialize_state(user_input: str, user_id: str) -> PlanExecute:
    memory = load_summary(user_id)
    return {
        "input": user_input,
        "plan": [],
        "past_steps": set(),
        "nodes_calls": set(),
        "next": "",
        "response": "",
        "visualization": "",
        "language": "",
        "translation": "",
        "automl_results": "",
        "last_memory": memory,
        "attached_img": "",
        "metadata": {},
    }
