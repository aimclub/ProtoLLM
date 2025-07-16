from langchain_core.output_parsers import PydanticOutputParser

from protollm.agents.agent_utils.pydantic_models import (Chat, Plan,
                                                         ReplanAction, Worker)

chat_parser = PydanticOutputParser(pydantic_object=Chat)
planner_parser = PydanticOutputParser(pydantic_object=Plan)
supervisor_parser = PydanticOutputParser(pydantic_object=Worker)
replanner_parser = PydanticOutputParser(pydantic_object=ReplanAction)
