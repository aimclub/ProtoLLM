import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from protollm.connectors import create_llm_connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def basic_call_example(url_with_name: str):
    """
    Example of using a model to get an answer.

    Args:
        url_with_name: Model URL combined with the model name
    """
    try:
        model = create_llm_connector(url_with_name, temperature=0.015, top_p=0.95)
        res = model.invoke("Tell me a joke")
        logging.info(res.content)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    

# Some models do not support explicit function calls, so the system prompt will be used for this. If it is not
# specified, it will be generated from the tool description and response format. If specified, it will be
# supplemented.
def function_call_example_with_functions(url_with_name: str):
    """
    Example of using a function to create a connector for function calls. Tools are defined as functions with the
    @tool decorator from LangChain.

    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)
    mssgs = [
        SystemMessage(content=""),
        HumanMessage(content="Build a plan for placing new schools with a budget of 5 billion rubles."),
    ]

    @tool
    def territory_by_budget(is_best_one: bool, budget: int | None, service_type: str) -> str:
        """
        Get potential territories for building a new service of a given type, considering the budget (amount in
        rubles).  This function should be used if the discussion involves placement, creation, construction, or erection
        of new services, including parks. Possible service types are strictly in the following list: ['school',
        'clinic', 'kindergarten', 'park']. The budget is optional. If the user does not specify a budget, the parameter
        will remain empty (None). Do not set default values yourself.
        Required arguments: ["service_type"]

        Args:
            is_best_one (bool): Flag indicating whether to select the best territory.
            budget (int | None): Budget amount in rubles.
            service_type (str): The new service being planned for construction. Possible service types are strictly in
                the following list: ['school', 'clinic', 'kindergarten', 'park']. Select the type that the user is
                interested in constructing.

        Returns:
            str: Analysis result.
        """
        return f"The best territory for {service_type} with a budget of {budget} has been found."

    @tool
    def parks_by_budget(budget: int | None) -> str:
        """
        Get parks suitable for improvement, considering the specified budget (amount in rubles). This function is used
        only if the discussion involves improving existing parks, not creating new ones. The budget is optional. If the
        user does not specify a budget, the parameter will remain empty (None). Do not set default values yourself.
        Required arguments: []

        Args:
            budget (int | None): Budget amount in rubles.

        Returns:
            str: Analysis result.
        """
        return f"Parks for improvement with a budget of {budget} have been found."

    tools_as_functions = [territory_by_budget, parks_by_budget]
    model_with_tools = model.bind_tools(tools=tools_as_functions, tool_choice="auto")
    try:
        res = model_with_tools.invoke(mssgs)
        logging.info(res.content)
        logging.info(res.tool_calls)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def function_call_example_with_dicts(url_with_name: str):
    """
    Example of using a function to create a connector for function calls. Tools are defined as dictionaries.

    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)
    mssgs = [
        SystemMessage(content=""),
        HumanMessage(content="Build a plan for placing new schools with a budget of 5 billion rubles."),
    ]

    tools_as_dicts = [
        {
            "name": "territory_by_budget",
            "description": (
                "Get potential territories for building a new service of a given type, considering the budget "
                "(amount in rubles). This function should be used if the discussion involves placement, creation, "
                "construction, or erection of new services, including parks. Possible service types are strictly in "
                "the following list: ['school', 'clinic', 'kindergarten', 'park']. The budget is optional. If the user "
                "does not specify a budget, the parameter will remain empty (None). Do not set default values yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "is_best_one": {
                        "type": "boolean",
                        "description": (
                            "Flag indicating whether to select the best territory."
                        ),
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Budget amount in rubles.",
                    },
                    "service_type": {
                        "type": "string",
                        "description": (
                            "The new service being planned for construction. Possible service types are strictly in "
                            "the following list: ['school', 'clinic', 'kindergarten', 'park']. Select the type that "
                            "the user is interested in constructing."
                        ),
                    },
                },
                "required": ["service_type"],
            },
        },
        {
            "name": "parks_by_budget",
            "description": (
                "Get parks suitable for improvement, considering the specified budget (amount in rubles). "
                "This function is used only if the discussion involves improving existing parks, not creating new ones. "
                "The budget is optional. If the user does not specify a budget, the parameter will remain empty (None). "
                "Do not set default values yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "integer",
                        "description": "Budget amount. May be specified in the request. Default value is None.",
                    },
                },
                "required": [],
            },
        },
    ]

    model_with_tools = model.bind_tools(tools=tools_as_dicts, tool_choice="auto")
    try:
        res = model_with_tools.invoke(mssgs)
        logging.info(res.content)
        logging.info(res.tool_calls)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def structured_output_example_with_dict(url_with_name: str):
    """
    Example of using a model to produce a structured response with a dictionary schema.

    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)

    Joke = {
        "title": "joke",
        "description": "Joke to tell user.",
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "The setup of the joke",
            },
            "punchline": {
                "type": "string",
                "description": "The punchline to the joke",
            },
            "rating": {
                "type": "integer",
                "description": "How funny the joke is, from 1 to 10",
                "default": None,
            },
        },
        "required": ["setup", "punchline"],
    }

    structured_model = model.with_structured_output(schema=Joke)
    try:
        res = structured_model.invoke("Tell me a joke about cats")
        logging.info(res)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def structured_output_example_with_pydantic(url_with_name: str):
    """
    Example of using a model to produce a structured response with a Pydantic class schema.

    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)

    class Joke(BaseModel):
        """Joke to tell user."""
        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )

    structured_model = model.with_structured_output(schema=Joke)
    try:
        res = structured_model.invoke("Tell me a joke about cats")
        logging.info(res)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    load_dotenv("../config.env") # Change path to your config file if needed or pass URL with name directly
    
    # model_url_and_name = os.getenv("LLAMA_URL")
    # model_url_and_name = os.getenv("GIGACHAT_URL")
    # model_url_and_name = os.getenv("DEEPSEEK_URL")
    # model_url_and_name = os.getenv("DEEPSEEK_R1_URL")
    # model_url_and_name = os.getenv("GPT4_URL")
    # model_url_and_name = os.getenv("OPENAI_URL")
    # model_url_and_name = os.getenv("OLLAMA_URL")
    model_url_and_name = os.getenv("SELF_HOSTED_LLM")
    
    # Uncomment the example you want to run
    basic_call_example(model_url_and_name)
    function_call_example_with_functions(model_url_and_name)
    function_call_example_with_dicts(model_url_and_name)
    structured_output_example_with_dict(model_url_and_name)
    structured_output_example_with_pydantic(model_url_and_name)
    