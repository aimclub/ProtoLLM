"""
Example of using an agent in a chemical pipeline with tools for drug generation (by API).

Process:
- Reading from a file with example queries.
- Pipeline starts.
- Results are written to the same file.
"""

from langchain.agents import create_structured_chat_agent, AgentExecutor, tool
import requests
import json
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from examples.real_world.chemical_pipeline.validate_tools import (
    validate_decompose,
    compute_metrics,
    validate_conductor,
)


def make_markdown_table(props: dict) -> str:
    """Create a table in Markdown format dynamically based on dict keys.

    Args:
        props (dict): properties of molecules

    Returns:
        str: table with properties
    """
    # get all the keys for column headers
    headers = list(props.keys())

    # prepare the header row
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # get the number of rows (assuming all lists in the dictionary are the same length)
    num_rows = len(next(iter(props.values())))

    # fill the table rows dynamically based on the keys
    for i in range(num_rows):
        row = [str(props[key][i]) for key in headers]
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table


# Define tools using the @tool decorator
@tool
def request_mols_generation(num: int) -> list:
    """Generates random molecules.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": num, "cuda": True, "mean_": 0.0, "std_": 1.0, "case_": "RNDM"}
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_alzheimer(num: int) -> list:
    """Generation of drug molecules for the treatment of Alzheimer's disease. GSK-3beta inhibitors with high activity. \
    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Alzhmr",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_multiple_sclerosis(num: int) -> list:
    """Generation of molecules for the treatment of multiple sclerosis.\
            There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
            BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Sklrz",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_dyslipidemia(num: int) -> list:
    """
    Generation of molecules for the treatment of dyslipidemia.
    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and
    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances
    reverse cholesterol transport via ABCA1 upregulation
    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be
    PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.

    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Dslpdm",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_acquired_drug_resistance(num: int) -> list:
    """
    Generation of molecules for acquired drug resistance.
    Molecules that selectively induce apoptosis in drug-resistant tumor cells.
    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "TBLET",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_lung_cancer(num: int) -> list:
    """
    Generation of molecules for the treatment of lung cancer.
    Molecules are inhibitors of KRAS protein with G12C mutation.
    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.
    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D,
    V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.
    """
    params = {"numb_mol": num, "cuda": True, "mean_": 0.0, "std_": 1.0, "case_": "Cnsr"}
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_parkinson(num: int) -> list:
    """
    Generation of molecules for parkinson.
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Prkns",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


# List of tools
tools = [
    gen_mols_parkinson,
    gen_mols_lung_cancer,
    gen_mols_acquired_drug_resistance,
    gen_mols_dyslipidemia,
    gen_mols_multiple_sclerosis,
    gen_mols_alzheimer,
    request_mols_generation,
]

# Create the system and human prompts
system_prompt = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a JSON blob to specify a tool by providing an "action" key (tool name) and an "action_input" key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per JSON blob, as shown:

{{ "action": $TOOL_NAME, "action_input": $INPUT }}

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action: $JSON_BLOB

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action: {{ "action": "Final Answer", "action_input": "Final response to human" }}


Begin! Reminder to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. 
Respond directly if appropriate. Format is Action:```$JSON_BLOB``` then Observation
In the "Final Answer" you must ALWAYS display all generated molecules!!!
For example answer must consist table (!):
| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |
\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| Fc1ccc2c(c1)CCc1ccccc1-2 | 0.6064732613170888 
| 1.721973678244476 | 0 | 0 | 0 | 0 | 1 | 0 |\n| O=C(Nc1ccc(C(=O)c2ccccc2)cc1)c1ccc(F)cc1 | 0.728441789442482 
| 1.4782662488060723 | 0 | 0 | 0 | 0 | 1 | 1 |\n| O=C(Nc1ccccc1)c1ccc(NS(=O)(=O)c2ccc3c(c2)CCC3=O)cc1 | 
0.6727786031171711 | 1.9616124655434675 | 0 | 0 | 0 | 0 | 0 | 0 |\n| Cc1ccc(C)c(-n2c(=O)c3ccccc3n(Cc3ccccc3)c2=O)c1 
| 0.5601042919484651 | 1.920664623176684 | 0 | 0 | 0 | 0 | 1 | 1 |\n| Cc1ccc2c(c1)N(C(=O)CN1C(=O)NC3(CCCc4ccccc43)C1=O)CC2 
| 0.8031696199670261 | 3.3073398307371438 | 0 | 0 | 0 | 1 | 1 | 0 |"
"""

human_prompt = """{input}
{agent_scratchpad}
(Reminder to respond in a JSON blob no matter what)"""

system_message = SystemMessagePromptTemplate.from_template(
    system_prompt,
    input_variables=["tools", "tool_names"],
)
human_message = HumanMessagePromptTemplate.from_template(
    human_prompt,
    input_variables=["input", "agent_scratchpad"],
)

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        human_message,
    ]
)

# Initialize the custom LLM
llm = Llama31ChatModel(
    api_key="API_KEY_VSE_GPT",
    base_url="https://api.vsegpt.ru/v1",
    model="meta-llama/llama-3.1-70b-instruct",
    temperature=0.5,
    max_tokens=5000,
)

# Create the structured chat agent
agent = create_structured_chat_agent(
    llm=llm, tools=tools, prompt=prompt, stop_sequence=True
)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    output_keys=["output"],
    early_stopping_method="generate",
)

# Example usage of the agent
if __name__ == "__main__":
    path = "examples/chemical_pipeline/queries_responses_chemical.xlsx"
    questions = pd.read_excel(path).values.tolist()

    for i, q in enumerate(questions):
        print("Task № ", i)
        response = agent_executor.invoke({"input": q[1]})

        validate_decompose(i, response["intermediate_steps"], path)
        for n, tools_pred in enumerate(response["intermediate_steps"]):
            name_tool = tools_pred[0].tool
            func = {"name": name_tool}
            validate_conductor(i, func, n, path)

        # Access the output
        final_answer = response["output"]
        # Print the final answer
        print(f"Agent's Response: \n {final_answer}")
