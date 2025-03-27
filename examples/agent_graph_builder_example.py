"""Example of building a multi-agent system with a scenario agent using GraphBuilder"""

import os
from typing import Annotated, Optional

import pubchempy as pcp
import rdkit.Chem as Chem
import requests
from ChemCoScientist.agents.agents_prompts import worker_prompt
from langchain.tools.render import render_text_description
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from rdkit.Chem.Descriptors import CalcMolDescriptors

from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector


@tool
def calc_prop_tool(
    smiles: Annotated[str, "The SMILES of a molecule"],
    property: Annotated[str, "The property to predict."],
):
    """Use this to predict molecular property.
    Can calculate refractive index and freezing point
    Do not call this tool more than once.
    Do not call another tool if this returns results."""
    result = 44.09
    result_str = f"Successfully calculated:\n\n{property}\n\nStdout: {result}"
    return result_str


@tool
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """Use this to convert molecule name to smiles format. Only use for organic molecules"""
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            compound = pcp.get_compounds(mol, "name")
            smiles = compound[0].canonical_smiles
            return smiles
        except BaseException as e:
            # logger.exception(f"'name2smiles' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't obtain smiles, the name is wrong"


@tool
def smiles2name(smiles: Annotated[str, "SMILES of a molecule"]):
    """Use this to convert SMILES to IUPAC name of given molecule"""

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                iupac_name = data["PropertyTable"]["Properties"][0]["IUPACName"]
                return iupac_name
            else:
                return "I've couldn't get iupac name"

        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't get iupac name"


@tool
def smiles2prop(
    smiles: Annotated[str, "SMILES of a molecule"], iupac: Optional[str] = None
):
    """Use this to calculate all available properties of given molecule. Only use for organic molecules
    params:
    smiles: str, smiles of a molecule,
    iupac: optional, default is None, iupac of molecule"""

    try:
        if iupac:
            compound = pcp.get_compounds(iupac, "name")
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


@tool
def generate_molecule(
    params: Annotated[str, "Description of target molecule"], config: RunnableConfig
):
    """Use this to generate a molecule with given description. Returns smiles. Only use for organic molecules"""
    llm = config["configurable"].get("model")
    try:
        prompt = (
            "Generate smiles of molecule with given description. Answer only with smiles, nothing more: \
            Question: The molecule is a nitrogen mustard drug indicated for use in the treatment of chronic lymphocytic leukemia (CLL) and indolent B-cell non-Hodgkin lymphoma (NHL) that has progressed during or within six months of treatment with rituximab or a rituximab-containing regimen.  Bendamustine is a bifunctional mechlorethamine derivative capable of forming electrophilic alkyl groups that covalently bond to other molecules. Through this function as an alkylating agent, bendamustine causes intra- and inter-strand crosslinks between DNA bases resulting in cell death.  It is active against both active and quiescent cells, although the exact mechanism of action is unknown. \
            Answer: CN1C(CCCC(=O)O)=NC2=CC(N(CCCl)CCCl)=CC=C21 \
            Question: The molecule is a mannosylinositol phosphorylceramide compound having a tetracosanoyl group amide-linked to a C20 phytosphingosine base, with hydroxylation at C-2 and C-3 of the C24 very-long-chain fatty acid. It is functionally related to an Ins-1-P-Cer(t20:0/2,3-OH-24:0).\
            Answer: CCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)C(O)CCCCCCCCCCCCCCCC \
            Question: "
            + params
            + "\n Answer: "
        )
        res = llm.invoke(prompt)
        smiles = res.content
        return smiles
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


def chemist_node(state, config: dict):
    """
    Executes a chemistry-related task using a ReAct-based agent.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state containing the task plan.
    config : dict
        Configuration dictionary containing the LLM model and related settings.

    Returns
    -------
    Command
        An object specifying the next execution step and updates to the state.
    """
    llm = config["configurable"]["llm"]
    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config["configurable"]["state"] = state
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto="replan_node",
                update={
                    "past_steps": [(task, agent_response["messages"][-1].content)],
                    "nodes_calls": [("chemist_node", agent_response["messages"])],
                },
            )

        except Exception as e:
            print(
                f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"
        },
    )


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

    model = create_llm_connector(
        "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
    )

    chem_tools = [name2smiles, smiles2name, smiles2prop, generate_molecule]
    tools_rendered = render_text_description(chem_tools)
    chem_tools_rendered = tools_rendered

    conf = {
        "recursion_limit": 50,
        "configurable": {
            "llm": model,
            "max_retries": 1,
            "scenario_agents": ["chemist_node"],
            "scenario_agent_funcs": {"chemist_node": chemist_node},
            "tools_for_agents": {
                "chemist_node": [chem_tools_rendered],
            },
            "tools_descp": tools_rendered,
        },
    }
    graph = GraphBuilder(conf)

    res_1 = graph.run(
        {"input": "What is the name of the molecule with the SMILES 'CCO'?"}, debug=True
    )
    res_2 = graph.run({"input": "What can you do?"}, debug=True)
    res_3 = graph.run({"input": "Определи IUPAC для молекулы CCO"}, debug=True)
    res_4 = graph.run(
        {"input": "Сгенерируй какую-нибудь полезную молекулу для здоровья."}, debug=True
    )
