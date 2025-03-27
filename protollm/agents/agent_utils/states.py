from typing_extensions import TypedDict
from typing import Annotated, List, Tuple
import operator


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    next: str
    response: str
    visualization: str
    language: str
    translation: str
    automl_results: str
    nodes_calls: Annotated[List[Tuple], operator.add]
