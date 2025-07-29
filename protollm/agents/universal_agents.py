import copy
import json
import time
from typing import Annotated, Dict, List, Union

from langchain_core.exceptions import OutputParserException
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from protollm.agents.agent_prompts import (
    build_planner_prompt,
    build_replanner_prompt,
    build_supervisor_prompt,
    chat_prompt,
    summary_prompt,
    worker_prompt,
)
from protollm.agents.agent_utils.parsers import (
    chat_parser,
    planner_parser,
    replanner_parser,
    supervisor_parser,
)
from protollm.agents.agent_utils.pydantic_models import Plan, ReplanAction, Response
from protollm.tools.web_tools import web_tools_rendered

# TODO: make real embedder, not dummy
store = InMemoryStore(index={"embed": lambda x: [[1.0, 2.0] for _ in x], "dims": 2})


def subgraph_start_node(state, config):
    print("Start node")
    return state


def subgraph_end_node(state, config):
    return state


def web_search_node(
    state: dict, config: dict
) :
    """
    Executes a web search task using a language model (LLM) and predefined web tools.

    Parameters
    ----------
    state : dict
        Dictionary representing the current execution state, expected to contain:
            - "plan" (List[str]): A list of steps to be executed by the agent.
    config : dict
        Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of the language model used for reasoning and task execution.
            - 'max_retries' (int): The maximum number of retry attempts if the web search fails.
            - 'web_tools' (List[BaseTool]): A list of predefined web tools to be used by the agent (can be empty).
    Returns
    -------
    Command 
        An object that contains either the next step for execution or an error response if retries are exhausted.
 
    Notes 
    -----
    - If web tools are not provided, the function creates an agent without them.
    - The function attempts to perform the task from the first step of the plan.
    - Retries are handled with exponential backoff on failure.
    - If all attempts fail, returns a fallback response.
    """
    llm = config["configurable"]["llm"]
    max_retries = config["configurable"]["max_retries"]

    if "web_tools" in config["configurable"].keys():
        web_tools = config["configurable"]["web_tools"]
    else:
        from protollm.tools.web_tools import web_tools

    web_agent = create_react_agent(llm, web_tools or [], prompt=worker_prompt)
    task = state["task"]

    for attempt in range(max_retries):
        try:
            agent_response = web_agent.invoke(
                {"messages": [("user", task + " You must search!")]}
            )
            for i, m in enumerate(agent_response["messages"]):
                if m.content == []:
                    agent_response["messages"][i].content = ""
            return Command(
                update={
                    "past_steps": Annotated[set, "or_"](
                        {(task, agent_response["messages"][-1].content)}
                    ),
                    "nodes_calls": Annotated[set, "or_"](
                        {
                            (
                                "web_search",
                                tuple(
                                    (m.type, m.content)
                                    for m in agent_response["messages"]
                                ),
                            )
                        }
                    ),
                }
            )
        except Exception as e:
            print(
                f"Web Search failed: {str(e)}. Retrying ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)


def supervisor_node(state: Dict[str, Union[str, List[str]]], config: dict) -> Command:
    """
    Oversees the execution of a given plan by formulating the next task for an agent and handling
    responses via an LLM-based supervisor.

    Parameters
    ----------
    state : dict
        Dictionary representing the current execution state, expected to contain:
            - "plan" (List[str]): A list of steps the supervisor will help execute.
            - "input" (str, optional): Initial user input or request.
    config : dict
        Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of a language model used by the supervisor.
            - 'max_retries' (int): Maximum number of retry attempts in case of errors.
            - 'scenario_agents' (list): List of agents/tools and their descriptions for prompt building.
            - 'tools_for_agents' (dict): Mapping of tools available to each agent.
    Returns
    -------
    Command
        A command with instructions for the next step or a fallback response message.

    Raises
    ------
    Exception
        Handles API call errors by applying exponential backoff on retries.

    Notes
    -----
    - Forms the task based on the first step of the provided plan.
    - If no plan or input is available, prompts the user to rephrase their request.
    - If all retries fail, returns a fallback message suggesting alternative assistance.
    """
    llm = config["configurable"]["llm"]
    max_retries = config["configurable"]["max_retries"]
    scenario_agents = config["configurable"]["scenario_agents"]
    tools_for_agents = config["configurable"]["tools_for_agents"]

    config["configurable"]["tools_for_agents"]["web_search"] = [web_tools_rendered]

    plan = state.get("plan")

    if not plan and not state.get("input"):
        return {
            "response": "I can't answer your question right now. Maybe I can assist with something else?",
            "end": True,
        }

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    supervisor_chain = (
        build_supervisor_prompt(scenario_agents, tools_for_agents)
        | llm
        | supervisor_parser
    )

    for attempt in range(max_retries):
        try:
            from protollm.agents.agent_utils.states import PlanExecute

            response = supervisor_chain.invoke({"input": [("user", task_formatted)]})
            subgraph = StateGraph(PlanExecute)
            subgraph.add_node("subgraph_start_node", subgraph_start_node)
            subgraph.add_node("subgraph_end_node", subgraph_end_node)
            subgraph.add_edge(START, "subgraph_start_node")
            added_nodes = []

            for i, node_name in enumerate(response.next):
                # get task for current agent from plan
                task_for_agent = None
                if isinstance(task, list) and i < len(task):
                    task_for_agent = task[i]
                else:
                    task_for_agent = task

                if node_name in added_nodes:
                    import random

                    node_name_new = node_name + str(random.randint(1000, 10000))
                    node_copy = copy.deepcopy(
                        config["configurable"]["scenario_agent_funcs"][node_name]
                    )

                    # Wrap the agent function to add needed task to state
                    def wrapped_agent_func(s, func=node_copy, task=task_for_agent):
                        "Wrappe"
                        s = s.copy()
                        s["task"] = task
                        return func(s, config)

                    subgraph.add_node(node_name_new, wrapped_agent_func)
                    subgraph.add_edge("subgraph_start_node", node_name_new)
                    subgraph.add_edge(node_name_new, "subgraph_end_node")
                else:
                    agent_func = config["configurable"]["scenario_agent_funcs"][
                        node_name
                    ]

                    def wrapped_agent_func(s, func=agent_func, task=task_for_agent):
                        s = s.copy()
                        s["task"] = task
                        return func(s, config)

                    subgraph.add_node(node_name, wrapped_agent_func)
                    subgraph.add_edge("subgraph_start_node", node_name)
                    subgraph.add_edge(node_name, "subgraph_end_node")

                added_nodes.append(node_name)
            subgraph.add_edge("subgraph_end_node", END)
            subgraph = subgraph.compile()

            res = subgraph.invoke(state, config)
            return res

        except Exception as e:
            print(
                f"Supervisor error: {str(e)}. Retrying attempt ({attempt + 1}/{max_retries})"
            )
            time.sleep(2**attempt)  # Exponential backoff

    state[
        "response"
    ] = "I can't answer your question right now. Maybe I can assist with something else?"
    state["end"] = True
    return state


def format_plan(plan: List[Dict[str, List[str]]]) -> str:
    """Make plan for correctly displaying in replanner prompt"""
    if not plan:
        return "No plan"

    result = ""
    for i, k in enumerate(plan):
        result += f"Step № {i + 1}: " + str(k)
    return result


def plan_node(
    state: Dict[str, Union[str, List[Dict]]], config: dict
) -> Union[Dict[str, List[Dict]], Command]:
    """
    Generates an execution plan using a language model (LLM) based on the provided input.
    """
    llm = config["configurable"]["llm"]
    max_retries = config["configurable"]["max_retries"]
    tools_descp = config["configurable"]["tools_descp"]
    adds_prompt = config["configurable"]["prompts"]["planner"]
    last_memory = state.get("last_memory", "")

    planner = (
        build_planner_prompt(tools_descp, last_memory, additional_hints=adds_prompt)
        | llm
        | planner_parser
    )

    query = state["input"]

    for attempt in range(max_retries):
        try:
            plan = planner.invoke({"input": query})
            state["plan"] = plan.steps
            return state

        except OutputParserException as e:
            # try to extract error
            try:
                error_str = str(e)
                start_idx = error_str.find('{"steps"')
                end_idx = error_str.rfind("}") + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = error_str[start_idx:end_idx]
                    partial_output = json.loads(json_str)

                    # Make plan from succces part of response
                    plan = Plan(steps=partial_output["steps"])
                    state["plan"] = plan.steps
                    return state
            except:
                print(
                    f"Failed to recover from parser error. Retry ({attempt + 1}/{max_retries})"
                )

        except Exception as e:
            print(f"Planner failed: {e}. Retry ({attempt + 1}/{max_retries})")
            time.sleep(2**attempt)

    return Command(
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        }
    )


def replan_node(
    state: Dict[str, Union[str, List[Dict]]], config: dict
) -> Union[Dict[str, Union[List[Dict], str]], Command]:
    """
    Refines or adjusts an existing execution plan based on previous steps and current state.
    """
    llm = config["configurable"]["llm"]
    max_retries = config["configurable"]["max_retries"]
    tools_descp = config["configurable"]["tools_descp"]
    try:
        adds_prompt = config["configurable"]["prompts"]["replanner"]
    except:
        adds_prompt = ""
    last_memory = state.get("last_memory", "")

    replanner = (
        build_replanner_prompt(tools_descp, last_memory, adds_prompt)
        | llm
        | replanner_parser
    )

    query = state["input"]
    current_plan = state.get("plan", [])
    past_steps = set(state.get("past_steps", []))

    # state['past_steps'] = set(state.get("past_steps", []))

    for attempt in range(max_retries):
        try:
            formatted_plan = format_plan(current_plan)
            formatted_past = str([i for i in set(past_steps)])

            output = replanner.invoke(
                {"input": query, "plan": formatted_plan, "past_steps": formatted_past}
            )

            if output.action == "response":
                state["response"] = output.response
                return state
            else:
                state["plan"] = output.steps or []
                state["next"] = 'supervisor'
                return state

        except OutputParserException as e:
            try:
                error_str = str(e)
                start_idx = error_str.find('{"action"')
                end_idx = error_str.rfind("}") + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = error_str[start_idx:end_idx]
                    partial_output = json.loads(json_str)

                    action = ReplanAction(**partial_output)

                    if action.action == "response":
                        state["response"] = action.response
                    else:
                        state["plan"] = action.steps or []
                    return state
            except:
                print(
                    f"Failed to recover from parser error. Retry ({attempt + 1}/{max_retries})"
                )

        except Exception as e:
            print(f"Replanner failed: {e}. Retry ({attempt + 1}/{max_retries})")
            time.sleep(2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I'm having trouble processing your request. Could you try asking differently?"
        },
    )


def summary_node(
    state: Dict[str, Union[str, List[str]]], config: dict
) -> Union[Dict[str, str], Command]:
    """
    Summarizes the system's response based on the provided input query and past steps.

    Parameters
    ----------
    state : dict
        Contains keys 'response', 'input', 'past_steps'
    config : dict
        Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of the language model used for generating summaries.
            - 'max_retries' (int): The maximum number of attempts to retry the summary generation in case of errors.

    Returns
    -------
    dict
        Dictionary with a summarized response under the key 'response'.
    Command
        Fallback response if summary generation fails after all retries.

    Notes
    -----
    - Uses summary_prompt and the language model to create summaries.
    """
    llm = config["configurable"]["llm"]
    max_retries = config["configurable"]["max_retries"]
    adds_prompt = config["configurable"]["prompts"]["summary"]

    system_response = state["response"]
    query = state["input"]
    past_steps = state["past_steps"]

    summary_agent = summary_prompt + "\n" + adds_prompt | llm

    for attempt in range(max_retries):
        try:
            output = summary_agent.invoke(
                {
                    "query": query,
                    "system_response": system_response,
                    "intermediate_thoughts": past_steps,
                }
            )
            state["response"] = output.content
            return state

        except Exception as e:
            print(
                f"Summary generation failed: {e}. Retry ({attempt + 1}/{max_retries})"
            )
            time.sleep(2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        },
    )


def chat_node(state, config: dict):
    """
    Processes user input through a chat agent and returns an appropriate response
    or next action. This agent decides whether it can handle the user query itself.
    If yes, responds with the {"response": agent_answer}.
    Otherwise, calls main agentic system.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, containing "input" (the user message) and
    config : dict
        Configuration dictionary containing a "configurable" sub-dictionary with the LLM model
        under the key "model".

    Returns
    -------
    dict
        If the response is a direct reply, returns {"response": message, "visualization": None}.
        If the response requires an action, returns {"next": action, "visualization": None}.
        If retries are exhausted, transitions to the planner with an empty response.

    Raises
    ------
    Exception
        Handles errors related to API failures, implementing exponential backoff (`2 ** attempt`).

    Notes
    -----
    - Resets visualization state on new responses.
    """
    llm = config["configurable"]["llm"]
    chat_agent = chat_prompt | llm | chat_parser
    input = state["input"]
    max_retries = 1

    for attempt in range(max_retries):
        try:
            output = chat_agent.invoke(
                {
                    "input": input,
                    "last_memory": state["last_memory"],
                    "additional_hints_for_scenario_agents": config["configurable"][
                        "prompts"
                    ]["chat"],
                }
            )

            if isinstance(output.action, Response):
                state["response"] = output.action.response
                return state
            else:
                state["next"] = output.action.next
            state["visualization"] = None

        except Exception as e:  # Handle OpenAI API errors
            print(
                f"Chat failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)

    state["response"] = None
    return state
