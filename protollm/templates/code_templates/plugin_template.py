import json

import requests
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.tools import tool

from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel


@tool
def tool_example(var1: int) -> dict:
    """
    External tool.
    """
    params = {
        "var1": var1,
    }
    resp = requests.post("external_url", data=json.dumps(params))

    ans = json.loads(resp.json())

    return ans


tools = [tool_example]

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
In the "Final Answer" you must ALWAYS do required action!!!
For example answer must consist table (!):
<table examples here>
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
    api_key="API_KEY_HERE",
    base_url="URL_HERE",
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

response = agent_executor.invoke({"input": "test"})
