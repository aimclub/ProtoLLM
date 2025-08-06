from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from protollm.agents.agent_utils.parsers import (chat_parser, planner_parser,
                                                 replanner_parser,
                                                 supervisor_parser)


def build_planner_prompt(
    tools_rendered: str,
    last_memory: str,
    n_steps: int = 5,
    additional_hints=None,
    problem_statement=None,
    rules=None,
    desc_restrictions=None,
    examples=None,
    image_description=None,
) -> ChatPromptTemplate:
    image_description = image_description or "No attached."
    problem_statement = (
        problem_statement
        or """
    For the given objective, create a step-by-step plan. Each step should contain:
        - just one task in step: tasks that must be executed in order
        - 2 and more tasks in step: tasks that can be executed simultaneously
    """
    )
    rules = (
        rules
        or """
        1. Use just one task in step for dependent tasks
        2. Use 2 and more tasks in step for independent tasks
        3. Each step should contain at least one task
    """
    )
    desc_restrictions = (
        desc_restrictions or f"The number of steps should be no more than {n_steps}"
    )
    examples = (
        examples
        or """   
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
        }}"""
    )

    template = """
        {problem_statement}

        Rules:
        {rules}
        
        {desc_restrictions}

        Examples:
        {examples}

        Available tools: {tools_rendered}
        Previous context: {last_memory}
        Additional hints: {additional_hints}
        Decripton of attached image: {image_description}
        User request: {input}

        {format_instructions}
        """
    return ChatPromptTemplate.from_messages(
        [("system", template), ("human", "{input}")]
    ).partial(
        problem_statement=problem_statement,
        rules=rules,
        desc_restrictions=desc_restrictions,
        examples=examples,
        tools_rendered=tools_rendered,
        last_memory=last_memory,
        additional_hints=additional_hints,
        image_description=image_description,
        format_instructions=planner_parser.get_format_instructions(),
    )


def build_replanner_prompt(
    tools_rendered: str,
    last_memory: str,
    problem_statement=None,
    rules=None,
    examples=None,
    additional_hint=None,
) -> ChatPromptTemplate:
    problem_statement = (
        problem_statement
        or """You are a replanning expert. Your job is to adjust the original step-by-step plan based on completed tasks."""
    )
    rules = (
        rules
        or """
    1. Remove completed tasks from the plan.
    2. Add new steps only if required.
    3. If all tasks are completed, return a final response.
    4. Limit the total number of steps to 4.
    5. Maintain the step format:
        - Each step is a list of tasks
        - Use a single task per step for sequential execution
        - Use multiple tasks per step for parallel execution"""
    )
    examples = (
        examples
        or """
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
    """
    )
    additional_hint = additional_hint or "You must return only JSON!!! No text before!"
    return ChatPromptTemplate.from_template(
        """
    {problem_statement}
    
    Objective: {input}

    Original plan:
    {plan}

    Completed steps (remove these from plan):
    {past_steps}

    Update the plan according to the following rules:
    {rules}
    
    Examples:

    {examples}

    Context:
    Previous memory: {last_memory}
    Available tools: {tools_rendered}
    Additional hints: {additional_hint}

    {format_instructions}
    """
    ).partial(
        problem_statement=problem_statement,
        rules=rules,
        last_memory=last_memory,
        examples=examples,
        additional_hint=additional_hint,
        tools_rendered=tools_rendered,
        format_instructions=replanner_parser.get_format_instructions(),
    )


def build_supervisor_prompt(
    scenario_agents: list = ["web_search", "chemist", "nanoparticles", "automl"],
    tools_for_agents: dict = {"web_search": ["TavilySearchResults"]},
    last_memory: str = "",
    problem_statement=None,
    problem_statement_continue=None,
    rules=None,
    additional_rules=None,
    examples=None,
    enhancemen_significance=None,
) -> ChatPromptTemplate:
    tools_descp_for_agents = ""
    for agent, tools in tools_for_agents.items():
        tools_descp_for_agents += f"- {agent} has tools: {', '.join(tools)}\n"

    problem_statement = (
        problem_statement
        or "You are a supervisor managing a team of specialized workers. "
    )
    problem_statement_continue = (
        problem_statement_continue
        or "Your task is to assign workers to execute the next step in a multi-step plan.\n\n"
    )
    rules = rules or (
        "1. If the step contains only one task, assign one appropriate worker.\n"
        "2. If the step contains multiple parallel tasks (i.e. list of tasks), assign the same number of appropriate workers.\n"
        "   - Each task must have one worker\n"
        "   - All assigned workers must be able to handle their respective tasks based on their tools and capabilities.\n"
        "3. If no workers are needed (all tasks complete), return an empty list.\n"
        "4. Consider worker capabilities and tools:\n"
    )
    additional_rules = additional_rules or (
        "Output MUST be in JSON format with the key 'next' containing a list of workers.\n"
        "The number of workers MUST match the number of tasks in the current step.\n\n"
    )
    examples = examples or (
        '{{"next": ["a_worker", "b_worker", "c_worker"]}}\n'
        '{{"next": ["web_search", "chemist"]}}\n'
        '{{"next": ["automl"]}}\n'
        '{{"next": []}}\n\n'
    )
    enhancemen_significance = (
        enhancemen_significance or "You must follow the format!!! Always return json!\n"
    )

    supervisor_system_prompt = (
        problem_statement
        + f"Available workers: {', '.join(scenario_agents)}\n\n"
        + problem_statement_continue
        + "Rules:\n"
        + rules
        + f"{tools_descp_for_agents}\n\n"
        + additional_rules
        + "Example outputs:\n"
        + examples
        + enhancemen_significance
        + "Previous conversation context:\n"
        f"{last_memory}\n\n"
        "{format_instructions}"
        "User request: {input}"
    )

    return ChatPromptTemplate.from_messages(
        [("system", supervisor_system_prompt)]
    ).partial(format_instructions=supervisor_parser.get_format_instructions())


worker_prompt = "You are a helpful assistant. You can use provided tools. \
    If there is no appropriate tool, or you can't use one, answer yourself"


def build_summary_prompt(additional_hints="", problem_statement=None, rules=None):
    problem_statement = (
        problem_statement
        or """Your task is to formulate final answer based on system_response using 
        intermediate_thoughts to make sure user gets full answer to their query."""
    )
    rules = (
        rules
        or """Your response must be direct answer to user query, don't write too much text. 
        You can use phrases like: it have been done, etc. Instead you must directly 
        tell what have been done, 
        extract all important results
        You should respond in markdown format. MAKE SURE YOUR RESPONSE IS THE ANSWER TO THE USER QUERY
        You must double check that your respond is the answer to user query."""
    )

    summary_prompt = ChatPromptTemplate.from_template(
        """{problem_statement}
        {rules}

        Your objective is this:
        User query: {query};
        System_response: {system_response};
        intermediate_thoughts: {intermediate_thoughts};
        """
        + additional_hints
    ).partial(problem_statement=problem_statement, rules=rules)
    return summary_prompt


def build_vision_prompt(problem_statement=None):
    problem_statement = (
        problem_statement
        or """
Analyze this image exhaustively following this structured framework. Prioritize precision, objectivity, and completeness in your description.

1. Core Content
Primary subject(s):
Identify and describe all dominant objects/entities (e.g., "A 35-year-old Caucasian woman wearing a red wool coat").
Specify exact counts, spatial arrangements, and relative sizes (e.g., "Three golden retrievers positioned in a triangle formation around a fire hydrant").

Actions/Events:
Describe observable activities (e.g., "A child is blowing bubbles while running toward a swing set").
Note implied narratives (e.g., "A half-eaten cake suggests an interrupted celebration").

2. Visual Elements
Composition:
Rule of thirds, symmetry, or leading lines.
Depth indicators (foreground/midground/background layers).

Colors & Lighting:
Hex codes or precise color names (e.g., "Pantone 19-4052 Classic Blue").
Light direction and quality (e.g., "Harsh overhead sunlight casting 45-degree shadows").

Text & Symbols:
Transcribe all text verbatim (e.g., "Sign reads 'CAUTION: Wet Floor' in bold Helvetica").
Decode symbols/logos (e.g., "Nike Swoosh logo on left shoe").

3. Contextual Analysis
Temporal/Geographical Cues:
Time indicators (e.g., "Deciduous trees with autumn foliage suggest mid-October").
Architectural styles or license plates for location inference.

Cultural/Subcultural Markers:
Fashion trends, religious symbols, or subculture signifiers (e.g., "Patch on jacket depicts the anarchy symbol").

4. Technical Observations
Image Quality:
Resolution artifacts, noise levels, or blur gradients.
Suspected editing traces (e.g., "Clone stamp artifacts near the subject’s elbow").

Perspective & Optics:
Focal length estimation (e.g., "Wide-angle distortion evident in curved horizon").
Depth of field (e.g., "f/2.8 bokeh effect isolating the subject").

5. Anomalies & Uncertainties
Flag ambiguities explicitly:
"Unidentifiable object in upper-right corner (possibly a drone or bird).
Shadow under the table suggests a hidden object not visible in frame.Э

Start answer with 'In the picture I see...'
"""
    )
    return SystemMessage(content=problem_statement)


def build_chat_prompt(
    problem_statement=None, additional_hints=None, last_memory=None
) -> SystemMessage:
    problem_statement = (
        problem_statement
        or """
    Now, the given objective, check whether it is simple enough to answer yourself. \
    If you can answer without any help and tools and the question is simple inquiry, then write your answer. \
    If you can't do that, call next worker: planner. \
    You should't answer to a several-sentenced questions. You can only chat with user on simple topics.
    """
    )

    system = f"""
    Here is what the user and system previously discussed:
    {last_memory}

    {problem_statement}

    If a user asks about your capabilities, tell him something from this:
    {additional_hints}
    
    Your output should match this dict format, don't add any intros!!! It is important!
    {{
    "action": {{
        "next" | "response" : str | str
    }}
    }}\n\n
    {chat_parser.get_format_instructions()}
    """

    return SystemMessage(content=system)
