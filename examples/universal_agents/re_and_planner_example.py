"""
Example of launching the Agent-Planner and RePlanner
"""

from protollm.agents.universal_agents import plan_node, replan_node
from protollm.connectors import create_llm_connector

if __name__ == "__main__":
    tools_description = """name2smiles(mol: typing.Annotated[str, 'Name of a molecule']) 
    - Use this to convert molecule name to smiles format. 
    Only use for organic molecules\n
    
    smiles2name(smiles: typing.Annotated[str, 'SMILES of a molecule']) 
    - Use this to convert SMILES to IUPAC name of given molecule\n
    
    smiles2prop(smiles: typing.Annotated[str, 'SMILES of a molecule'], iupac: Optional[str] = None) 
    - Use this to calculate all available properties of given molecule. 
    Only use for organic molecules\n    
    params:\n    
    smiles: str, smiles of a molecule,\n    
    iupac: optional, default is None, iupac of molecule\n
    
    generate_molecule(params: typing.Annotated[str, 'Description of target molecule'], 
    config: langchain_core.runnables.config.RunnableConfig) 
    - Use this to generate a molecule with given description. 
    Returns smiles. Only use for organic molecules"""

    response = """
    Generate molecule with IC50 less then 5.
    """
    state = {"input": response, "language": "English"}

    model = create_llm_connector(
        "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
    )
    conf = {"llm": model, "max_retries": 1, "tools_descp": tools_description}
    res = plan_node(state, conf)

    fake_plan = [
        "1) Convert [Ca+2].[Cl-].[Cl-] to IUPAC name using smiles2name function with the given SMILES as input."
    ]
    state["plan"] = fake_plan
    state["past_steps"] = [(fake_plan[0], "Calcium chloride")]
    replan_res = replan_node(state, conf)

    print("Planner answer: ")
    print(res["plan"])

    print("\nReplanner answer with fake plan: ")
    print(replan_res["plan"])
