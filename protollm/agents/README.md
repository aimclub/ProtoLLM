# GraphBuilder

A tool for building an agent system based on a universal agents core.

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
            "agent_1",
            "agent_2",
            ...
        ],

        # nodes for scenario agents
        # must be implemented by analogy as universal and agents
        "scenario_agent_funcs": {
            "agent_1": agent1_node,
            ...
        },

        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            # here can be description of langchain web tools (not TavilySearch)
            # "web_serach": [web_tools_rendered],
            "agent_1": [description_for_agent_tools],
            ...
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

            ...
        },

        # These prompts will be added as hints in ProtoLLM
        # must be compiled for each system independently
        "prompts": {
            "planner": "...",
            "chat": """Here description about you system"""
        },
    },
}