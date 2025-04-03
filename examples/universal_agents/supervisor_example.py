"""
Example of launching the Supervisor agent
"""

from langchain_community.tools.tavily_search import TavilySearchResults
from protollm.agents.universal_agents import supervisor_node

from protollm.connectors import create_llm_connector

if __name__ == "__main__":
    response = """
    Find the molecule was discovered in 2008 by American scientists.
    """
    state = {
        "input": response,
        "language": "English",
        "plan": [
            "Use web_search to find the molecule was discovered in 2008 by American scientists."
        ],
    }

    model = create_llm_connector(
        "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
    )
    conf = {
        "llm": model,
        "max_retries": 1,
        "scenario_agents": ["web_search"],
        "tools_for_agents": {"web_serach": [TavilySearchResults]},
    }
    res = supervisor_node(state, conf)
    print("Next node will be: ", res.update["next"])
