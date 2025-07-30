import time
from typing import Annotated

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


def playground_scenario_node(state, config: dict) -> Command:
    print("--------------------------------")
    print("Playground agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")
    
    system_prompt = config["configurable"]["additional_agents_info"]["playground_scenario_node"]["system_prompt"]
    tools = config["configurable"]["additional_agents_info"]["playground_scenario_node"]["tools"]
    
    task = state["task"]
    plan = state["plan"]

    llm = config["configurable"]["llm"]
    chem_agent = create_react_agent(
        llm, tools, state_modifier=system_prompt
    )

    task_formatted = f"""For the following plan:
    {str(plan)}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config["configurable"]["state"] = state
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                update={
                    "past_steps": Annotated[set, "or_"](
                        {(task, agent_response["messages"][-1].content)}
                    ),
                    "nodes_calls": Annotated[set, "or_"](
                        {
                            (
                                "chemist_node",
                                tuple(
                                    (m.type, m.content)
                                    for m in agent_response["messages"]
                                ),
                            )
                        }
                    ),
                },
            )

        except Exception as e:
            print(f"Chemist failed: {str(e)}. Retrying ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)