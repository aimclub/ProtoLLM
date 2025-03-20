import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Union

import yaml
from ChemCoScientist.dataset_handler.chembl.chembl_utils import ChemblLoader
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from protollm.agents.agent_prompts import (
    build_planner_prompt,
    build_replanner_prompt,
    build_supervisor_prompt,
    retranslate_prompt,
    summary_prompt,
    translate_prompt,
    worker_prompt,
)
from protollm.agents.agents_utils.parsers import (
    planner_parser,
    replanner_parser,
    supervisor_parser,
    translator_parser,
)
from protollm.agents.agents_utils.pydantic_models import Response


def in_translator_node(state: dict, conf: dict) -> Union[Dict, Command]:
    """
    Detects the input language and translates it into English if necessary.

    Args:
        state (dict): The current execution state containing an 'input' key.
        conf (dict): Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of a language model used for translation.
            - 'max_retries' (int): The maximum number of attempts to retry translation in case of errors.
    Returns:
        dict: Updated state with 'language' and optionally 'translation' fields.
    """
    llm = conf["llm"]
    max_retries = conf["max_retries"]

    translator_agent = translate_prompt | llm | translator_parser
    query = state.get("input", "")

    for attempt in range(max_retries):
        try:
            output = translator_agent.invoke(query)
            state["language"] = output.language
            if output.language != "English":
                state["translation"] = output.translation
            return state
        except Exception as e:
            print(f"Translator error: {str(e)} (Attempt {attempt + 1}/{max_retries})")
            if "api key" in str(e).lower():
                state["response"] = "Invalid API key."
                return state
            if "404" in str(e):
                state["response"] = "LLM service unavailable. Check proxy settings."
                return state
            time.sleep(1.2**attempt)

    return Command(
        goto=END,
        update={"response": "Translation service failed after multiple attempts."},
    )


def re_translator_node(state: dict, conf: dict) -> Union[Dict, Command]:
    """
    Translates a system-generated response back into the user's language.

    Args:
        state (dict): Current execution state containing:
            - 'response' (str): The system-generated response in English.
            - 'language' (str): The user's original language code or name.
        conf (dict): Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of a language model used for back-translation.
            - 'max_retries' (int): The maximum number of retry attempts in case of errors.

    Returns:
        Command: Command object to update state or end execution.
    """
    llm = conf["llm"]
    max_retries = conf["max_retries"]
    language = state["language"]

    if language == "English":
        return state

    translator_agent = retranslate_prompt | llm
    query = state.get("response", "")

    for attempt in range(max_retries):
        try:
            translated = translator_agent.invoke({"input": query, "language": language})
            state["response"] = translated
            return state
        except Exception as e:
            print(
                f"Retranslation error: {str(e)} (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(1.2**attempt)

    return Command(
        goto=END,
        update={"response": "Unable to translate response after multiple attempts."},
    )


def supervisor_node(state: Dict[str, Union[str, List[str]]], conf: dict) -> Command:
    """
    Oversees the execution of a given plan by formulating the next task for an agent and handling
    responses via an LLM-based supervisor.

    Parameters
    ----------
    state : dict
        Dictionary representing the current execution state, expected to contain:
            - "plan" (List[str]): A list of steps the supervisor will help execute.
            - "input" (str, optional): Initial user input or request.
    conf : dict
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
    llm = conf["llm"]
    max_retries = conf["max_retries"]
    scenario_agents = conf["scenario_agents"]
    tools_for_agents = conf["tools_for_agents"]

    plan = state.get("plan")

    if not plan and not state.get("input"):
        return Command(
            goto=END,
            update={
                "response": "I'm sorry, I couldn't understand your request. Could you please rephrase it?"
            },
        )

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
            response = supervisor_chain.invoke({"input": [("user", task_formatted)]})

            return Command(goto=response.next, update={"next": response.next})
        except Exception as e:
            print(
                f"Supervisor error: {str(e)}. Retrying attempt ({attempt + 1}/{max_retries})"
            )
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        },
    )


def web_search_node(
    state: Dict[str, Union[str, List[str]]], conf: dict
) -> Union[Dict, Command]:
    """
    Executes a web search task using a language model (LLM) and predefined web tools.

    Parameters
    ----------
    state : dict
        Dictionary representing the current execution state, expected to contain:
            - "plan" (List[str]): A list of steps to be executed by the agent.
    conf : dict
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
    llm = conf["llm"]
    max_retries = conf["max_retries"]
    web_tools = conf["web_tools"]

    web_agent = create_react_agent(llm, web_tools or [], state_modifier=worker_prompt)

    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    for attempt in range(max_retries):
        try:
            agent_response = web_agent.invoke({"messages": [("user", task_formatted)]})
            state["past_steps"] = [(task, agent_response["messages"][-1].content)]
            state["nodes_calls"] = [("web_search_node", agent_response["messages"])]
            return state
        except Exception as e:
            print(
                f"Web search failed with error: {str(e)}. Retrying... ({attempt + 1}/{max_retries})"
            )
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        },
    )


def plan_node(
    state: Dict[str, Union[str, List[str]]], conf: dict
) -> Union[Dict[str, List[str]], Command]:
    """
    Generates an execution plan using a language model (LLM) based on the provided input.

    Parameters
    ----------
    state : dict
        The current execution state, expected to contain:
            - "input" (str): The user's original input request.
            - "language" (str): Detected language of the input.
            - "translation" (str, optional): The translated input if the language is not English.
    conf : dict
        Configuration dictionary containing:
            - 'llm' (BaseChatModel): An instance of the language model used to generate the plan.
            - 'max_retries' (int): Maximum number of retry attempts in case of failures.
            - 'tools_descp' (str): A description of available tools, included in the prompt to guide plan creation.

    Returns
    -------
    dict
        A dictionary with the generated plan under the key "plan".
    Command
        A fallback response if planning fails after all retries.

    Notes
    -----
    - Uses planner_prompt, llm, and planner_parser to create a structured execution plan.
    - Handles JSON parsing errors and attempts to recover partial outputs.
    - Applies exponential backoff between retries.
    """
    llm = conf["llm"]
    max_retries = conf["max_retries"]
    tools_descp = conf["tools_descp"]
    print(planner_parser)

    planner = build_planner_prompt(tools_descp) | llm | planner_parser
    query = state["input"] if state["language"] == "English" else state["translation"]

    for attempt in range(max_retries):
        try:
            plan = planner.invoke({"messages": [("user", query)]})
            return {"plan": plan.steps}

        except OutputParserException as e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[.*?\]\s*\}', str(e), re.DOTALL)
            if match:
                try:
                    structured_output = json.loads(match.group(0))
                    return {"plan": structured_output["steps"]}
                except json.JSONDecodeError as json_err:
                    print(
                        f"Planner JSON parse error: {json_err}. Retry ({attempt + 1}/{max_retries})"
                    )

        except Exception as e:
            print(f"Planner failed: {e}. Retry ({attempt + 1}/{max_retries})")
            time.sleep(2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        },
    )


def replan_node(
    state: Dict[str, Union[str, List[str]]], conf: dict
) -> Union[Dict[str, Union[List[str], str]], Command]:
    """
    Refines or adjusts an existing execution plan based on previous steps and current state.

    Parameters
    ----------
    state : dict
        The current execution state, expected to contain:
            - "input" (str): The user's request or query.
            - "language" (str): Detected language of the input.
            - "translation" (str, optional): Translated input if not in English.
            - "plan" (list of str): The current plan consisting of step descriptions.
            - "past_steps" (list of tuples): Previously executed steps and their responses.
    conf : dict
        Configuration dictionary containing:
            - 'llm' (BaseChatModel): The language model used for replanning.
            - 'max_retries' (int): Number of retries if execution fails.
            - 'tools_descp' (dict): Description of tools available to assist in replanning.

    Returns
    -------
    dict
        A dictionary with either an updated plan under "plan" or a direct response under "response".
    Command
        A fallback response if replanning fails after all retries.

    Notes
    -----
    - Uses replanner_prompt, llm, and replanner_parser to adjust plans.
    - Handles parsing errors and extracts structured JSON if needed.
    - Applies exponential backoff between retries.
    """
    llm = conf["llm"]
    max_retries = conf["max_retries"]
    tools_descp = conf["tools_descp"]
    replanner = build_replanner_prompt(tools_descp) | llm | replanner_parser

    query = state["input"] if state["language"] == "English" else state["translation"]

    for attempt in range(max_retries):
        try:
            output = replanner.invoke(
                {
                    "input": query,
                    "plan": state["plan"],
                    "past_steps": state["past_steps"],
                }
            )

            if hasattr(output.action, "response"):
                return {"response": output.action.response}
            return {"plan": output.action.steps}

        except OutputParserException as e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[.*?\]\s*\}', str(e), re.DOTALL)
            if match:
                try:
                    structured_output = json.loads(match.group(0))
                    return {"plan": structured_output["steps"]}
                except json.JSONDecodeError as json_err:
                    print(
                        f"Replanner JSON parse error: {json_err}. Retry ({attempt + 1}/{max_retries})"
                    )

        except Exception as e:
            print(f"Replanner failed: {e}. Retry ({attempt + 1}/{max_retries})")
            time.sleep(2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Maybe I can assist with something else?"
        },
    )


def should_end(state) -> str:
    """
    Determines the next step based on the presence of a response.

    This function decides whether execution should proceed to summarization
    or require further supervision.

    Parameters
    ----------
    state : PlanExecute
        The current execution state, potentially containing a generated response.

    Returns
    -------
    str
        `"summary"` if a response is available, otherwise `"supervisor"`.

    Notes
    -----
    - If the `"response"` key is present and non-empty, summarization is triggered.
    - If no response is available, the system proceeds to the supervisor node.
    """
    if "response" in state and state["response"]:
        return "summary"
    else:
        return "supervisor"


def summary_node(
    state: Dict[str, Union[str, List[str]]], conf: dict
) -> Union[Dict[str, str], Command]:
    """
    Summarizes the system's response based on the provided input query and past steps.

    Parameters
    ----------
    state : dict
        Contains keys 'response', 'input', 'past_steps', and optionally 'translation'.
    conf : dict
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
    llm = conf["llm"]
    max_retries = conf["max_retries"]

    system_response = state["response"]
    query = (
        state["input"]
        if state.get("language", "English") == "English"
        else state["translation"]
    )
    past_steps = state["past_steps"]

    summary_agent = summary_prompt | llm

    for attempt in range(max_retries):
        try:
            output = summary_agent.invoke(
                {
                    "query": query,
                    "system_response": system_response,
                    "intermediate_thoughts": past_steps,
                }
            )
            return {"response": output.content}

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
