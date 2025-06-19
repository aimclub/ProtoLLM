## 🧪 Example of config for GraphBuilder

```python
from protollm.connectors import create_llm_connector

model = create_llm_connector(
    "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
)

conf = {
    # maximum number of recursions
    "recursion_limit": 25,
    "configurable": {
        "user_id": "1",
        "llm": model,
        "max_retries": 1,

        # list with string-names of scenario agents
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
            "dataset_builder_agent",
            "coder_agent",
        ],

        # nodes for scenario agents
        # must be implemented by analogy as universal and agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
        },

        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            # here can be description of langchain web tools (not TavilySearch)
            # "web_serach": [web_tools_rendered],
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "dataset_builder_agent": [dataset_builder_agent_description],
            "coder_agent": [coder_agent_description],
            "ml_dl_agent": [automl_agent_description],
        },

        # full descripton for agents tools
        "tools_descp": tools_rendered,

        # set True if you want to use web search like black-box
        "web_search": True,

        # add a key with the agent node name if you need to pass something to it
        "additional_agents_info": {

            "dataset_builder_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": "OPENAI_API_KEY",
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },

            "coder_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": "OPENAI_API_KEY",
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },

            "ml_dl_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": "OPENAI_API_KEY"q,
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },
        },

        # These prompts will be added as hints in ProtoLLM
        # must be compiled for each system independently
        "prompts": {
            "planner": "Before you start training models, plan to check your data for garbage using a dataset_builder_agent",
            "chat": """You are a chemical agent system. You can do the following:
                    - train generative models (generate SMILES molecules), train predictive models (predict properties)
                    - prepare a dataset for training
                    - download data from chemical databases: ChemBL, BindingDB
                    - perform calculations with chemical python libraries
                    - solve problems of nanomaterial synthesis
                    - analyze chemical articles
                    """
        },
    },
}