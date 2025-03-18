import json
import os
import uuid

import requests


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
    return json.loads(response.text)['access_token']


# List of models that do NOT support calling functions out-of-the-box yet
models_without_function_calling = ["r1", "deepseek-chat-alt", "test_model"]

# List of models that do NOT support structured outputs out-of-the-box yet
models_without_structured_output = ["deepseek-chat", "deepseek-chat-alt", "llama-3.3-70b-instruct", "test_model"]