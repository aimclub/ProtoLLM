import json
import os
from typing import Any, Dict, List, Optional
import re
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import requests

from definitions import CONFIG_PATH


load_dotenv(CONFIG_PATH)


def get_access_token() -> str:
    """
    Gets the access token by the authorisation key specified in the config.
    The token is valid for 30 minutes.
    
    Returns:
        Access token for Gigachat API
    """
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    request_id = uuid.uuid4()
    authorization_key = os.getenv("AUTHORIZATION_KEY")

    payload = {
      'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': f'{request_id}',
      'Authorization': f'Basic {authorization_key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return eval(response.text)['access_token']


class CustomChatOpenAI(ChatOpenAI):
    """
    A class that extends the ChatOpenAI base class to allow use with the LLama family of models, as they do not return
    tool calls in the tool_calls field of the response, but instead write them as an HTML string in the content field.
    """
    def invoke(self, *args, **kwargs) -> AIMessage:
        response = super().invoke(*args, **kwargs)
        
        if isinstance(response, AIMessage) and response.content.startswith("<function="):
            tool_calls = self._parse_function_calls(response.content)
            
            if tool_calls:
                response.tool_calls = tool_calls
                response.content = ""
        
        return response
    
    @staticmethod
    def _parse_function_calls(content: str) -> List[Dict[str, Any]]:
        """
        Parses LLM answer (HTML string) to extract function calls.
        
        Args:
            content: model response as an HTML string
            
        Returns:
            A list of dictionaries in tool_calls format/
        """
        tool_calls = []
        pattern = r"<function=(.*?)>(.*?)</function>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            function_name, function_args = match
            try:
                arguments = json.loads(function_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error when decoding function arguments: {e}")

            tool_call = {
                "id": f"call_{len(tool_calls) + 1}",
                "type": "tool_call",
                "name": function_name,
                "args": arguments
            }
            tool_calls.append(tool_call)
        
        return tool_calls



def create_llm_connector(model_url: str) -> CustomChatOpenAI | GigaChat:
    """Creates the proper connector for a given LLM service URL.

    Args:
        model_url: The LLM endpoint for making requests; should be in the format 'base_url;model_endpoint or name'
            - for vsegpt.ru service for example: 'https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct'
            - for Gigachat models family: 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions;Gigachat'
              for Gigachat model you should also install certificates from 'НУЦ Минцифры' -
              instructions - 'https://developers.sber.ru/docs/ru/gigachat/certificates'

    Returns:
        The ChatModel object from 'langchain' that can be used to make requests to the LLM service,
        use tools, get structured output.
    """
    if "vsegpt" in model_url:
        model_data = model_url.split(";")
        base_url, model_name = model_data[0], model_data[1]
        api_key = os.getenv("VSE_GPT_KEY")
        return CustomChatOpenAI(model_name=model_name, base_url=base_url, api_key=api_key)
    elif "gigachat":
        model_name = model_url.split(";")[1]
        access_token = get_access_token()
        return GigaChat(model=model_name, access_token=access_token)
    # Possible to add another LangChain compatible connector


if __name__ == "__main__":
    # Examples of use
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    
    # model = create_llm_connector(os.getenv("LLAMA_URL"))
    # model = create_llm_connector(os.getenv("GIGACHAT_URL"))
    model = create_llm_connector(os.getenv("GPT4_URL"))
    
    # Tool calling
    # messages = [
    #     SystemMessage(
    #         content=""
    #     ),
    #     HumanMessage(content="Построй мне план размещения новых школ с бюджетом на 5000000000 рублей"),
    # ]
    #
    #
    # @tool
    # def territory_by_budget(is_best_one: bool, budget: int | None, service_type: str) -> str:
    #     """
    #     Получить потенциальные территории для строительства нового сервиса заданного типа, с учетом бюджета.
    #
    #     Args:
    #         is_best_one (bool): Флаг, указывающий, нужно ли выбрать лучшую территорию
    #         budget (int | None): Размер бюджета в рублях
    #         service_type (str): Тип сервиса ('школа', 'поликлиника', 'детский сад', 'парк')
    #
    #     Returns:
    #         str: Результат анализа.
    #     """
    #     return f"Лучшая территория для {service_type} с бюджетом {budget} найдена."
    #
    #
    # @tool
    # def parks_by_budget(budget: int | None) -> str:
    #     """
    #     Получить парки, подходящие для благоустройства, с учетом заданного бюджета.
    #
    #     Args:
    #         budget (int | None): Размер бюджета в рублях.
    #
    #     Returns:
    #         str: Результат анализа.
    #     """
    #     return f"Парки для благоустройства с бюджетом {budget} найдены."
    #
    #
    # tools = [territory_by_budget, parks_by_budget]
    
    # model_with_tools = model.bind_tools(tools=tools, tool_choice="auto")
    # res = model_with_tools.invoke(messages)
    # print(res.tool_calls)
    
    # Structured output
    class Joke(BaseModel):
        """Joke to tell user."""
        
        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )
    
    structured_model = model.with_structured_output(schema=Joke)
    res = structured_model.invoke("Tell me a joke about cats")
    print(res)
    # Token usage can be seen as follows
    # print(res.response_metadata)
