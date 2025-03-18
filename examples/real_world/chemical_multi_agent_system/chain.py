"""
Example of multi-agents chemical pipeline with tools for drug generation (by API).
There are 2 agents.

Process:
- Reading from a file with example queries.
- Pipeline starts.
- Decomposer define tasks.
- Conductor-executor agent define and run tools, reflects on everyone tool response and return answer.
"""

from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from examples.real_world.chemical_multi_agent_system.tools import (
    gen_mols_parkinson,
    gen_mols_lung_cancer,
    gen_mols_acquired_drug_resistance,
    gen_mols_dyslipidemia,
    gen_mols_multiple_sclerosis,
    gen_mols_alzheimer,
    request_mols_generation,
)
from examples.real_world.chemical_multi_agent_system.prompting import (
    system_prompt_conductor,
    system_prompt_decomposer,
    human_prompt,
)


tools = [
    gen_mols_parkinson,
    gen_mols_lung_cancer,
    gen_mols_acquired_drug_resistance,
    gen_mols_dyslipidemia,
    gen_mols_multiple_sclerosis,
    gen_mols_alzheimer,
    request_mols_generation,
]


class Chain:
    def __init__(self, key: str):
        self.llm = Llama31ChatModel(
            api_key=key,
            base_url="https://api.vsegpt.ru/v1",
            model="meta-llama/llama-3.1-70b-instruct",
            temperature=0.5,
            max_tokens=5000,
        )
        self.agents_meta = {
            "conductor": {
                "prompt": system_prompt_conductor,
                "variables": ["tools", "tool_names"],
            },
            "decomposer": {"prompt": system_prompt_decomposer, "variables": []},
        }
        self.conductor = self._create_agent_executor("conductor")
        self.decomposer = self.llm

    def _complite_prompt_from_template(
        self, text: str, input_variables: list = ["tools", "tool_names"]
    ):
        system_message = SystemMessagePromptTemplate.from_template(
            text, input_variables=input_variables
        )
        human_message = HumanMessagePromptTemplate.from_template(
            human_prompt, input_variables=["input", "agent_scratchpad"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                human_message,
            ]
        )
        return prompt

    def _complite_prompt_no_template(self, system_txt: str, user_txt: str):
        messages = [SystemMessage(content=system_txt), HumanMessage(content=user_txt)]
        return messages

    def _create_agent_executor(self, agent_name: str):
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=tools,
            prompt=self._complite_prompt_from_template(
                self.agents_meta[agent_name]["prompt"],
                self.agents_meta[agent_name]["variables"],
            ),
            stop_sequence=True,
        )
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            output_keys=["output"],
            early_stopping_method="generate",
        )
        return executor

    def run_chain(self, user_msg: str):
        input = self._complite_prompt_no_template(
            self.agents_meta["decomposer"]["prompt"], user_msg
        )
        tasks = eval(eval(self.decomposer.invoke(input).content)["action_input"])
        for n, task in enumerate(tasks):
            answer = self.conductor.invoke({"input": task})["output"]
            print("Answer, part №: ", n + 1)
            print(answer)


# run pipeline on test data
if __name__ == "__main__":
    path = "examples/chemical_pipeline/queries_responses_chemical.xlsx"
    questions = pd.read_excel(path).values.tolist()
    chain = Chain(key="KEY_HERE")

    for i, q in enumerate(questions):
        print("Task № ", i)
        chain.run_chain(q[1])
