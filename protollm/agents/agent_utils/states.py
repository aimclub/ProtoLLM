import operator
from typing import Annotated, List, Tuple

from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]

    past_steps: Annotated[List[Tuple], operator.add]
    nodes_calls: Annotated[List[Tuple], operator.add]

    next: str
    response: str
    visualization: str
    language: str
    translation: str
    automl_results: str
    last_memory: str
    metadata: Annotated[dict, operator.or_]

    parallel_tasks: dict


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
        "past_steps": [],
        "next": "",
        "response": "",
        "visualization": "",
        "language": "",
        "translation": "",
        "automl_results": "",
        "nodes_calls": [],
        "last_memory": memory,
        "metadata": {},
    }
