import requests
import json

from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from langchain_core.tools import tool

from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from examples.real_world.chemical_pipeline.validate_tools import (
    validate_decompose,
    compute_metrics,
    validate_conductor,
)

# Create the system and human prompts
system_prompt = """
Respond to the human as helpfully and accurately as possible.
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
agent = create_structured_chat_agent(llm=llm, prompt=prompt, stop_sequence=True)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    verbose=True,
    return_intermediate_steps=True,
    output_keys=["output"],
    early_stopping_method="generate",
)

response = agent_executor.invoke({"input": "test"})
