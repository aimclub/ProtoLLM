from langchain_core.output_parsers import PydanticOutputParser

from protollm.agents.agents_utils.pydantic_models import (
    Act,
    Chat,
    Plan,
    Translation,
    Worker,
)

chat_parser = PydanticOutputParser(pydantic_object=Chat)
planner_parser = PydanticOutputParser(pydantic_object=Plan)
supervisor_parser = PydanticOutputParser(pydantic_object=Worker)
replanner_parser = PydanticOutputParser(pydantic_object=Act)
translator_parser = PydanticOutputParser(pydantic_object=Translation)
