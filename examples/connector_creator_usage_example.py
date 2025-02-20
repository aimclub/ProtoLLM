import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from protollm.definitions import CONFIG_PATH
from protollm.connectors.connector_creator import create_llm_connector


def basic_call_example(url_with_name: str):
    """
    Example of using a model to get an answer.
    
    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name, temperature=0.015, top_p=0.95)
    res = model.invoke("Tell me a joke")
    print(res.content)
    

def function_call_example(url_with_name: str):
    """
    Example of using a function to create a connector for function calls. Tools can be defined as functions with the
    @tool decorator from LangChain or as dictionaries.
    Some models do not support explicit function calls, so the system prompt will be used for this. If it is not
    specified, it will be generated from the tool description and response format. If specified, it will be
    supplemented.
    
    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)
    mssgs = [
        SystemMessage(
            content=""
        ),
        HumanMessage(content="Построй мне план размещения новых школ с бюджетом на 5000000000 рублей"),
    ]
    
    @tool
    def territory_by_budget(is_best_one: bool, budget: int | None, service_type: str) -> str:
        """
        Получить потенциальные территории для строительства нового сервиса заданного типа, с учетом бюджета.

        Args:
            is_best_one (bool): Флаг, указывающий, нужно ли выбрать лучшую территорию
            budget (int | None): Размер бюджета в рублях
            service_type (str): Тип сервиса ('школа', 'поликлиника', 'детский сад', 'парк')

        Returns:
            str: Результат анализа.
        """
        return f"Лучшая территория для {service_type} с бюджетом {budget} найдена."
    
    @tool
    def parks_by_budget(budget: int | None) -> str:
        """
        Получить парки, подходящие для благоустройства, с учетом заданного бюджета.

        Args:
            budget (int | None): Размер бюджета в рублях.

        Returns:
            str: Результат анализа.
        """
        return f"Парки для благоустройства с бюджетом {budget} найдены."

    tools_as_functions = [territory_by_budget, parks_by_budget]
    
    # tools_as_dicts = [
    #     {
    #         "name": "territory_by_budget",
    #         "description": (
    #             "Получить потенциальные территории для строительства нового сервиса заданного типа, с учетом бюджета"
    #             " (сумма в рублях). Эта функция должна использоваться, если речь идет о размещении, создании,"
    #             " строительстве или возведении новых сервисов, включая парки. Возможные значения типов сервисов строго"
    #             " в следующем списке: ['школа', 'поликлиника', 'детский сад', 'парк']."
    #             " Бюджет - опциональный параметр. Если пользователь не укажет бюджет, параметр останется пустым (None)."
    #             " Не задавайте значения по умолчанию сами."
    #             " Примеры: 1. Можно ли построить поликлиники в районе озера Долгого, если на строительство выделен"
    #             " бюджет 5496000000 рублей? -> budget=5496000000, service_type='поликлиника'; 2. Хватит ли выделенного"
    #             " бюджета в 4456567234 рублей на возведение парков в Пушкинский район СПБ? -> budget=45646217234,"
    #             " service_type='парк'; 3. На строительство парка на Пискаревке выделен бюджет 2563560030 рублей. Хватит"
    #             " ли этих денег? -> budget=45646217234, service_type='парк'; 4. Где мне разместить школу? ->"
    #             " is_best_one=True, service_type='школа'."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "is_best_one": {
    #                     "type": "string",
    #                     "description": (
    #                         "Флаг, в котором необходимо установить True, если пользователь просит лучшую территорию для"
    #                         " размещения сервиса или просит подобрать только один квартал для сервиса. В остальных"
    #                         " случаях - False."
    #                     ),
    #                 },
    #                 "budget": {
    #                     "type": "integer",
    #                     "description": "Размер бюджета. Может быть указан в запросе. Значение по умолчанию - None.",
    #                 },
    #                 "service_type": {
    #                     "type": "string",
    #                     "description": "Новый сервис, который собираются строить. Возможные значения типов сервисов"
    #                                    " строго в следующем списке: ['школа', 'поликлиника', 'детский сад', 'парк']."
    #                                    " Нужно подобрать тип, строительство которого интересует пользователя.",
    #                 },
    #             },
    #             "required": ["service_type"],
    #         },
    #     },
    #     {
    #         "name": "parks_by_budget",
    #         "description": (
    #             "Получить парки, подходящие для благоустройства, с учетом заданного бюджета (сумма в рублях)."
    #             " Эта функция используется только если речь идет о благоустройстве существующих парков, а не о создании"
    #             " новых."
    #             " Бюджет - опциональный параметр. Если пользователь не укажет бюджет, параметр останется пустым (None)."
    #             " Не задавайте значения по умолчанию сами."
    #             " Примеры: 1. На благоустройство парков в Академическом выделен бюджет 12345517000 рублей. Хватит ли"
    #             " этих денег? -> budget=12345517000: 2. Хватит ли мне 2846621000 рублей, чтобы благоустроить парк в"
    #             " Петергофе? -> budget=2846621000."
    #         ),
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "budget": {
    #                     "type": "integer",
    #                     "description": "Размер бюджета. Может быть указан в запросе. Значение по умолчанию - None.",
    #                 },
    #             },
    #             "required": [],
    #         },
    #     },
    # ]
    
    model_with_tools = model.bind_tools(tools=tools_as_functions, tool_choice="auto")
    res = model_with_tools.invoke(mssgs)
    print(res.content)
    print(res.tool_calls)


def structured_output_example(url_with_name: str):
    """
    An example of using a model to produce a structured response.
    
    Args:
        url_with_name: Model URL combined with the model name
    """
    model = create_llm_connector(url_with_name)

    # from pydantic import BaseModel, Field
    # from typing import Optional
    # class Joke(BaseModel):
    #     """Joke to tell user."""
    #     setup: str = Field(description="The setup of the joke")
    #     punchline: str = Field(description="The punchline to the joke")
    #     rating: Optional[int] = Field(
    #         default=None, description="How funny the joke is, from 1 to 10"
    #     )
    
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
    res = structured_model.invoke("Tell me a joke about cats")
    print(res)


if __name__ == "__main__":
    load_dotenv(CONFIG_PATH) # Change path to your config file if needed or pass URL with name directly
    
    # model_url_and_name = os.getenv("LLAMA_URL")
    # model_url_and_name = os.getenv("GIGACHAT_URL")
    model_url_and_name = os.getenv("DEEPSEEK_URL")
    # model_url_and_name = os.getenv("DEEPSEEK_R1_URL")
    # model_url_and_name = os.getenv("GPT4_URL")
    
    basic_call_example(model_url_and_name)
    # function_call_example(model_url_and_name)
    # structured_output_example(model_url_and_name)
    