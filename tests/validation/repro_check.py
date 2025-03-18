from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from protollm_tools.sdk.protollm_sdk.jobs.job import Job
from protollm_tools.sdk.protollm_sdk.jobs.job_context import JobContext

from protollm_tools.llm_agents_api.models import StairsLLMAgentResult
from protollm_tools.llm_agents_api.config import (
    CUSTOM_USER_MESSAGE,
    CUSTOM_SYSTEM_MESSAGE,
)
from protollm_tools.llm_agents_api.parse_result import parse_intermediate_steps
from protollm_tools.llm_agents_api.tools import (
    query_database_rag,
    get_time,
    get_resource,
    restore_works_edges,
    start_schedule,
    extract_scheduling_params,
)
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel


class StairsLLMAgentJob(Job):
    """
    Job class representing the main execution of the Stairs LLM Agent.

    Attributes:
        tools (list): List of tools available to the agent.
        prompt (ChatPromptTemplate): Chat prompt template used to structure the conversation.
        llm (Llama31ChatModel): Language model instance used by the agent for generating responses.
    """

    def __init__(self):
        """Initializes the StairsLLMAgentJob with tools, prompt, and language model."""
        super().__init__()
        self.tools = [
            query_database_rag,
            get_time,
            get_resource,
            restore_works_edges,
            start_schedule,
            extract_scheduling_params,
        ]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    CUSTOM_SYSTEM_MESSAGE,
                    input_variables=["tools", "tool_names"],
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessagePromptTemplate.from_template(
                    CUSTOM_USER_MESSAGE,
                    input_variables=["input", "agent_scratchpad"],
                ),
            ]
        )

        self.llm = Llama31ChatModel()

    def run(self, job_id: str, ctx: JobContext, **kwargs):
        """Executes the agent job, generating and saving the result.

        Args:
            job_id (str): Unique identifier for the job execution.
            ctx (JobContext): Context object for managing the job's lifecycle and data storage.
            **kwargs: Additional parameters which includes:
                request (str): The user query string.
                is_project (bool): Flag indicating if the request is a project.
                is_scheduling (bool): Flag indicating if the request is for scheduling.

        Returns:
            None. Saves the job result directly into the provided job context.
        """
        request: str = kwargs.get("request", "")
        is_project: bool or None = kwargs.get("is_project", None)
        request += "" if is_project is None else f" is_project={is_project}"
        is_scheduling: bool or None = kwargs.get("is_scheduling", None)
        request += "" if is_scheduling is None else f" is_scheduling={is_scheduling}"

        agent_request = {
            "input": request
        }  # , "context": {"is_project": is_project, "is_scheduling": is_scheduling}}

        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            stop_sequence=True,
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            output_keys=["output"],
        )
        result = agent_executor.invoke(agent_request)

        parsed_result = parse_intermediate_steps(result)
        dumped_result = parsed_result.model_dump()
        ctx.result_storage.save_dict(job_id, dumped_result)
        print("written to redis: ", parsed_result)


import uuid

from protollm_tools.sdk.protollm_sdk.jobs.utility import construct_job_context
from protollm_tools.sdk.protollm_sdk.utils.reddis import get_reddis_wrapper, load_result

from protollm_tools.llm_agents_api.jobs import StairsLLMAgentJob


def prepare_test_data(
    request: str, is_project: str or None, is_scheduling: str or None, tools: list[str]
) -> dict:
    return {
        "request": request,
        "is_project": is_project,
        "is_scheduling": is_scheduling,
        "tools": tools,
    }


SCHEDULE_QUERY = "Запусти планирования для проекта"

SCHEDULE_TEST = prepare_test_data(SCHEDULE_QUERY, True, False, ["start_schedule"])


def main():
    job_id = str(uuid.uuid4())
    ctx = construct_job_context("agent")
    job = StairsLLMAgentJob()
    request, is_project, is_scheduling, _ = SCHEDULE_TEST
    job.run(
        job_id=job_id, ctx=ctx, request=request, is_project=True, is_scheduling=False
    )
    rd = get_reddis_wrapper()
    result = str(load_result(rd, job_id, "agent"))


if __name__ == "__main__":
    main()
