"""
Example of launching the Agent-translator.
A phrase in Italian is fed to the input (it can be fed in any language).
Phrase will be translated into English (for execution within the intended system), and when
passed to the second agent, a response will be returned in the original language.
"""

from protollm.agents.universal_agents import in_translator_node, re_translator_node
from protollm.connectors import create_llm_connector

if __name__ == "__main__":
    state = {"input": "Ciao! Come stai?"}

    model = create_llm_connector(
        "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
    )
    conf = {"llm": model, "max_retries": 1}
    res = in_translator_node(state, conf)
    print("Language is: ", res["language"])
    print("Translate: ", res["translation"])

    res["response"] = "Made up answer..."
    total_res = re_translator_node(res, conf)

    print("Total answer: ")
    print(total_res["response"].content)
