from langgraph.graph import END, START, StateGraph

from protollm.agents.agent_utils.states import PlanExecute, initialize_state
from protollm.agents.universal_agents import (chat_node, plan_node,
                                              replan_node, summary_node,
                                              supervisor_node, web_search_node)
import copy
import time
from typing import Annotated

from langgraph.types import Command
from langgraph.prebuilt import create_react_agent


class GraphBuilder:
    """Builds a graph based on the basic structure of universal agents.
    Need to add your own scenario agents via 'conf'.

     Args:
        conf (dict): Configuration dictionary with the following structure:
            - recursion_limit (int): Maximum recursion depth for processing.
            - configurable (dict): Configurations for the agents and tools.
                - llm: BaseChatModel
                - max_retries (int): Number of retries for failed tasks.
                - scenario_agents (list): List of scenario agent names.
                - scenario_agent_funcs (dict): Mapping of agent names to their function (link on ready agent-node).
                - tools_for_agents (dict): Description of tools available for each agent.
                - tools_descp: Rendered descriptions of tools.

    Example:
        conf = {
            "recursion_limit": 50,
            "configurable": {
                "llm": model,
                "max_retries": 1,
                "scenario_agents": ["chemist_node"],
                "scenario_agent_funcs": {"chemist_node": chemist_node},
                "tools_for_agents": {
                    "chemist_node": [chem_tools_rendered]
                },
                "tools_descp": tools_rendered,
            }
        }
    """

    def __init__(self, conf: dict):
        self.conf = conf
        self.app = self._build()

    def _should_end_chat(self, state) -> str:
        """
        Determines whether to continue the chat or transition to a different process.

        Parameters
        ----------
        state : dict | TypedDict
            The current execution state, expected to contain "response".

        Returns
        -------
        str
            Returns "summary" if a response exists, otherwise "planner".

        Notes
        -----
        - This function helps decide whether further processing is needed.
        """
        if "response" in state and state["response"]:
            return "summary"
        else:
            return "planner"

    def _should_end(self, state) -> str:
        """
        Determines the next step based on the presence of a response.

        This function decides whether execution should proceed to summarization
        or require further supervision.

        Parameters
        ----------
        state : PlanExecute
            The current execution state, potentially containing a generated response.

        Returns
        -------
        str
            `"summary"` if a response is available, otherwise `"supervisor"`.

        Notes
        -----
        - If the `"response"` key is present and non-empty, summarization is triggered.
        - If no response is available, the system proceeds to the supervisor node.
        """
        if "response" in state and state["response"]:
            return "summary"
        else:
            return "supervisor"

    def _routing_function_supervisor(self, state):
        """Determines the next agent after Supervisor"""
        if state.get("end", False):
            return END
        return "replan_node"

    def _routing_function_planner(self, state):
        if state.get("response"):
            return END
        return "supervisor"
    

    def create_scenario_node(self, node_name: str):
        """This function creates a scenario node for the agent."""
        
        def scenario_node(state: dict, config: dict) -> Command:
            print(f"--------------------------------")
            print(f"{node_name} agent called")  
            print("Current task:")
            print(state["task"])
            print("--------------------------------")

            task = state["task"]
            plan = state["plan"]
            system_prompt = config["configurable"]["scenario_agents_info"][node_name]["system_prompt"]
            tools = config["configurable"]["scenario_agents_info"][node_name]["tools"]
            
            llm = config["configurable"]["llm"]
            
            agent = create_react_agent( 
                llm, tools, prompt=system_prompt  
            )

            task_formatted = f"""For the following plan:
            {str(plan)}\n\nYou are tasked with executing: {task}."""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    config["configurable"]["state"] = state
                    agent_response = agent.invoke({"messages": [("user", task_formatted)]})

                    return Command(
                        update={
                            "past_steps": Annotated[set, "or_"](
                                {(task, agent_response["messages"][-1].content)}
                            ),
                            "nodes_calls": Annotated[set, "or_"](
                                {
                                    (
                                        node_name,  
                                        tuple(
                                            (m.type, m.content)
                                            for m in agent_response["messages"]
                                        ),
                                    )
                                }
                            ),
                        },
                    )

                except Exception as e:
                    print(f"{node_name} failed: {str(e)}. Retrying ({attempt+1}/{max_retries})")
                    time.sleep(1.2**attempt)
        return scenario_node

    def _build(self):
        """Build graph based on a non-dynamic agent skeleton"""
        workflow = StateGraph(PlanExecute)
        workflow.add_node("chat", chat_node)
        workflow.add_node("planner", plan_node)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("replan_node", replan_node)
        workflow.add_node("summary", summary_node)

        if self.conf["configurable"]["web_search"]:
            workflow.add_node("web_search", web_search_node)
            workflow.add_edge("web_search", "replan_node")

        for agent_name in self.conf["configurable"][
            "scenario_agents"
        ]:
            node = copy.deepcopy(self.create_scenario_node(agent_name))
            workflow.add_node(agent_name, node)
            workflow.add_edge(agent_name, "replan_node")
            self.conf["configurable"]["scenario_agent_funcs"]={}
            self.conf["configurable"]["scenario_agent_funcs"][agent_name] = node

        workflow.add_edge(START, "chat")

        workflow.add_conditional_edges(
            "chat",
            self._should_end_chat,
            ["planner", "summary"],
        )
        workflow.add_conditional_edges(
            "planner",
            self._routing_function_planner,
            ["supervisor", END],
        )
        workflow.add_conditional_edges(
            "replan_node",
            self._should_end,
            ["supervisor", "summary"],
        )
        workflow.add_edge("summary", END)

        workflow.add_conditional_edges("supervisor", self._routing_function_supervisor)

        return workflow.compile()

    def stream(self, inputs: dict, user_id: str = "1"):
        """Start streaming the input through the graph."""
        inputs = initialize_state(user_input=inputs["input"], user_id=user_id)
        for event in self.app.stream(inputs, config=self.conf):
            for k, v in event.items():
                yield (v)
        try:
            print("\n\nFINALLY ANSWER: ", v["response"].content)
        except:
            print("\n\nFINALLY ANSWER: ", v["response"])
