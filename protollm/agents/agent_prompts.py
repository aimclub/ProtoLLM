from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from protollm.agents.agent_utils.parsers import (chat_parser, planner_parser,
                                                 replanner_parser,
                                                 supervisor_parser)


def build_planner_prompt(
    tools_rendered: str, last_memory: str, n_steps: int = 3, additional_hints: str = ""
) -> ChatPromptTemplate:
    template = """
        For the given objective, create a step-by-step plan. Each step should contain:
        - just one task in step: tasks that must be executed in order
        - 2 and more tasks in step: tasks that can be executed simultaneously

        Rules:
        1. Use just one task in step for dependent tasks
        2. Use 2 and more tasks in step for independent tasks
        3. Each step should contain at least one task

        Example:
        Request: "Prepare data, train model, then predict for molecule1 and molecule2"
        Response: {{
            "steps": [
                ["Prepare data"],
                ["Train model"],
                ["Predict for molecule1", "Predict for molecule2"]
            ]
        }}
        
        Example:
        Request: "Generate 5 molecules related to MEK1, make 3 molecules using the GSK model "
        Response: {{
            "steps": [
                ['Generate 5 molecules related to MEK1 using ml_dl_agent', 'Generate 3 molecules using the GSK model with ml_dl_agent']
            ]
        }}

        Available tools: {tools_rendered}
        Previous context: {last_memory}
        Additional hints: {additional_hints}

        User request: {input}

        {format_instructions}
        """
    return ChatPromptTemplate.from_messages(
        [("system", template), ("human", "{input}")]
    ).partial(
        tools_rendered=tools_rendered,
        last_memory=last_memory,
        additional_hints=additional_hints,
        format_instructions=planner_parser.get_format_instructions(),
    )


def build_replanner_prompt(
    tools_rendered: str, last_memory: str, n_steps: int = 3, adds_prompt: str = ""
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
    You are a replanning expert. Your job is to adjust the original step-by-step plan based on completed tasks.

    Objective: {input}

    Original plan:
    {plan}

    Completed steps (remove these from plan):
    {past_steps}

    Update the plan according to the following rules:
    1. Remove completed tasks from the plan.
    2. Add new steps only if required.
    3. If all tasks are completed, return a final response.
    4. Limit the total number of steps to {n_steps}.
    5. Maintain the step format:
        - Each step is a list of tasks
        - Use a single task per step for sequential execution
        - Use multiple tasks per step for parallel execution

    Examples:

    1. Final response (your answer):
    {{
        "action": "response",
        "response": "Your final answer here"
    }}

    2. Updated plan (your answer):
    {{
        "action": "steps",
        "steps": [
            ["Train model"],
            ["Predict for molecule1", "Predict for molecule2"]
        ]
    }}

    You must return only JSON!!! No text before!

    Context:
    Previous memory: {last_memory}
    Available tools: {tools_rendered}
    Additional hints: {adds_prompt}

    {format_instructions}
    """
    ).partial(
        n_steps=str(n_steps),
        last_memory=last_memory,
        tools_rendered=tools_rendered,
        adds_prompt=adds_prompt,
        format_instructions=replanner_parser.get_format_instructions(),
    )


def build_supervisor_prompt(
    scenario_agents: list = ["web_search", "chemist", "nanoparticles", "automl"],
    tools_for_agents: dict = {"web_search": ["TavilySearchResults"]},
    last_memory: str = "",
) -> ChatPromptTemplate:
    tools_descp_for_agents = ""
    for agent, tools in tools_for_agents.items():
        tools_descp_for_agents += f"- {agent} has tools: {', '.join(tools)}\n"

    supervisor_system_prompt = (
        "You are a supervisor managing a team of specialized workers. "
        f"Available workers: {', '.join(scenario_agents)}\n\n"
        "Your task is to assign workers to execute the next step in a multi-step plan.\n\n"
        "Rules:\n"
        "1. If the step contains only one task, assign one appropriate worker.\n"
        "2. If the step contains multiple parallel tasks (i.e. list of tasks), assign the same number of appropriate workers.\n"
        "   - Each task must have one worker\n"
        "   - All assigned workers must be able to handle their respective tasks based on their tools and capabilities.\n"
        "3. If no workers are needed (all tasks complete), return an empty list.\n"
        "4. Consider worker capabilities and tools:\n"
        f"{tools_descp_for_agents}\n\n"
        "Output MUST be in JSON format with the key 'next' containing a list of workers.\n"
        "The number of workers MUST match the number of tasks in the current step.\n\n"
        "Example outputs:\n"
        '{{"next": ["a_worker", "b_worker", "c_worker"]}}\n'
        '{{"next": ["web_search", "chemist"]}}\n'
        '{{"next": ["automl"]}}\n'
        '{{"next": []}}\n\n'
        "You must follow the format!!! Always return json!\n"
        "Previous conversation context:\n"
        f"{last_memory}\n\n"
        "User request: {input}"
    )

    return ChatPromptTemplate.from_messages(
        [("system", supervisor_system_prompt)]
    ).partial(format_instructions=supervisor_parser.get_format_instructions())


worker_prompt = "You are a helpful assistant. You can use provided tools. \
    If there is no appropriate tool, or you can't use one, answer yourself"


summary_prompt = ChatPromptTemplate.from_template(
    """Your task is to formulate final answer based on system_response using 
intermediate_thoughts to make sure user gets full answer to their query.
Your response must be direct answer to user query, don't write too much text. 
You can use phrases like: it have been done, etc. Instead you must directly 
tell what have been done, 
extract all important results
You should respond in markdown format. MAKE SURE YOUR RESPONSE IS THE ANSWER TO THE USER QUERY
You must double check that your respond is the answer to user query.


Your objective is this:
User query: {query};
System_response: {system_response};
intermediate_thoughts: {intermediate_thoughts};
"""
)

chat_prompt = ChatPromptTemplate.from_template(
    """
Here is what the user and system previously discussed:
{last_memory}

Now, the given objective, check whether it is simple enough to answer yourself. \
If you can answer without any help and tools and the question is simple inquery, then write your answer. If you can't do that, call next worker: planner
If the question is related to running models or checking for presence, training, inference - call planer!
You should't answer to a several-sentenced questions. You can only chat with user on a simle topics.

If a user asks about your capabilities, tell him something from this:
{additional_hints_for_scenario_agents}

Your objective is this:
{input}

Your output should match this JSON format, don't add any intros!!! It is important!
{{
  "action": {{
    "next" | "response" : str | str
  }}
}}
"""
).partial(format_instructions=chat_parser.get_format_instructions())
