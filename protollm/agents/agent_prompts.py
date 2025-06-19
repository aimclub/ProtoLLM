from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from protollm.agents.agent_utils.parsers import (
    planner_parser,
    replanner_parser,
    supervisor_parser,
    translator_parser,
    chat_parser
)


def build_planner_prompt(tools_rendered: str, last_memory: str, n_steps: int = 3, additional_hints_for_scenario_agents: str = '') -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                For the given objective, come up with a simple step by step plan how to answer 
                the question. You can't answer yourself. Don't write any answers, only parts of the plan \
                This plan should involve individual tasks, that if executed correctly 
                by other workers will yield 
                the correct answer. Do not add any superfluous steps. Don't make things up!!! Don't plan to process a dataset unless the user asks for it.
                The result of the final step should be the final answer. Make sure that each step has all 
                the information needed - do not skip steps. Do no more than """ + str(n_steps) + """.
                You must directly insert important information into your plan. 
                
                For example, if the task is: Prepare a dataset for training from a file so that the properties where docking score < -1 remain. 
                Run training of the generative model, name the case "docking".
                Your plan is: ["Prepare a dataset for training from a file so that the properties where docking score < -1 remain.", 
                Run training of the generative model, name the case "docking"]
                For example: Launch prediction using ml-model for IC50 value for molecule Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1. 
                Your plan is: Launch inferenct using automl agent to predict IC50 value for Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1. 
                For example: Run training of the generative model on the data '/Users/alina/Desktop/ITMO/ChemCoScientist/data_dir_for_coder/chembl_ic50_data.xlsx', name the case Docking.
                Your plan is: Run training by automl agent on the data '/Users/alina/Desktop/ITMO/ChemCoScientist/data_dir_for_coder/chembl_ic50_data.xlsx', name the case Docking.
                
                ONLY return JSON in this exact format: {{"steps": ["Step 1", "Step 2", "Step 3"]}}.
                Don't add any introduction. Don't add model training to your plan unless you're specifically asked to do so.
                
                Additional hints:
                """ + additional_hints_for_scenario_agents +
                """
                For better understanding you are provided with information about previous dialogue of the user and you:
                """+ last_memory + f"\nSystem has these tools and agents {tools_rendered}",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(format_instructions=planner_parser.get_format_instructions())


def build_replanner_prompt(tools_rendered: str, last_memory: str, n_steps: int = 3) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
        or the given objective, come up with a simple step by step plan how to answer 
        the question. You can't answer yourself \
        This plan should involve individual tasks, that if executed correctly by other workers 
        will yield the correct answer. Do not add any superfluous steps. \
        You can't refer to results of previous steps. Instead you must directly insert
        such results in your plan. If you see step number more than """ + str(n_steps) + """, you should generate final response \
        The result of the final step should be the final answer. 

        Your objective was this:
        {input}

        Your original plan was this (don't take too many steps! (no more than """ + str(n_steps) + """)):
        {plan}

        You have currently done the following steps:
        {past_steps}

        Update your plan accordingly. If no more steps are needed and you can return to 
        the user, then respond with final response, which answers the objective.
        Only add steps 
        to the plan that still NEED to be done. Do not return previously 
        done steps as part of the plan.
        If the answer to the user's question is already contained in the response, return the answer to the user. 
        Don't over-plan.
        Don't add model training to your plan unless you're specifically asked to do so.

                        
        For better understanding you are provided with information about previous dialogue of the user and you:
        """
        + last_memory + 
        f" Here are tools that system has: {tools_rendered}"
        + """
        Your output should match this JSON format, don't add any intros
        {{
        "action": {{
            "response" | "steps" : str | List[str] 
        }}
        }}
        """
    ).partial(format_instructions=replanner_parser.get_format_instructions())


def build_supervisor_prompt(
    scenario_agents: list = ["web_search", "chemist", "nanoparticles", "automl"],
    tools_for_agents: dict = {"web_search": [TavilySearchResults]},
    last_memory: str = ""
):
    tools_descp_for_agents = ""
    for agent, tools in tools_for_agents.items():
        tools_descp_for_agents += agent + "has these tools: " + str(tools) + "\n\n"
    supervisor_system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {scenario_agents}. Given the following user request, "
        "respond with the worker to act next. "
        'Your output must be json format: {{"next": "worker"}}'
        "Don't write any intros." + tools_descp_for_agents + 
        """
        For better understanding you are provided with information about previous dialogue of the user and you:
        """ + last_memory
    )
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [("system", supervisor_system_prompt), ("human", "{input}")]
    ).partial(format_instructions=supervisor_parser.get_format_instructions())
    return supervisor_prompt


worker_prompt = "You are a helpful assistant. You can use provided tools. \
    If there is no appropriate tool, or you can't use one, answer yourself" 

translate_prompt = ChatPromptTemplate.from_template(
    """For the given input determine it's language. \
If it is English do nothing, else write translation and language. 

Your objective is this:
{input}

Respond only with this JSON format. Don't add any intros
{{
    "translation": str,
    "language": str
}}
"""
).partial(format_instructions=translator_parser.get_format_instructions())

retranslate_prompt = ChatPromptTemplate.from_template(
    """Translate given input to given language. Don't 'translation to ...', 
just write the translation itself.
Don't translate specific termins and smiles formulas, and the word 'SMILES'

If the input contains a list of properties, format them as a Markdown table. Round numbers to three digits

### Example:
#### Input:
Smiles of water is O
Boiling point: 100°C  
Melting point: 0°C  
Density: 1 g/cm³  

#### Expected Output:
Smiles of water is O
It's properties are listed in the table below:
| Property        | Value    |
|----------------|---------|
| Boiling point  | 100°C   |
| Melting point  | 0°C     |
| Density        | 1 g/cm³ |

---

Your objective is this:
input: {input};
language: {language};
"""
)


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

