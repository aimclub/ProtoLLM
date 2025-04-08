# Connectors

Currently, the `create_llm_connector` function from the module [connector_creator.py](connector_creator.py) is used to
create connectors for LLMs. The base classes are classes of type ChatModel from the 
[LangChain](https://python.langchain.com/docs/introduction/) library. Accordingly, all basic methods (for example
`invoke`, `bind_tools`, `with_structured_output`) for calling models
are supported.

Not all models support tool calling and structured output by default. To circumvent this, an add-on/add-on of the
corresponding system prompt is applied. Lists of models that have problems are kept separately in the
[utils.py](utils.py) module. You free to extend these lists if needed.

It is also important to note that additional certifications are required to use Gigachat models. Instructions on how to
install them can be found [here](https://developers.sber.ru/docs/ru/gigachat/certificates).

## Supported providers/LLM hosting services:
1. https://api.vsepgt.ru/v1
   - VSE_GPT_KEY env variable
   - example of an argument for a function: `https://api.vsegpt.ru/v1;openai/gpt-4o-mini`
2. https://api.openai.com/v1
   - OPENAI_KEY env variable
   - example of an argument for a function: `https://api.openai.com/v1;gpt-4o-mini`
3. https://gigachat.devices.sberbank.ru/api/v1
   - AUTHORIZATION_KEY env variable, which can be obtained from your personal account
   - example of an argument for a function: `https://gigachat.devices.sberbank.ru/api/v1/chat/completions;GigaChat-Pro`
4. Ollama (no API key required)
   - example of an argument for a function: `ollama;http://localhost:11434;llama3.2`
5. Self-hosted LLM (under FastAPI, no API key required)
   - example of an argument for a function: `self_hosted;http://99.99.99.99:9999;example_model`

Before use, make sure that your config file has the necessary API key or set it in the environment yourself.

## Examples of usage

To create a connector it is necessary to pass to the function the URL of the corresponding service combined with the
model name with a semicolon (;), for example: `https://api.vsegpt.ru/v1;openai/gpt-4o-mini`

It is also possible to pass additional parameters for the model.  Available parameters:
- `temperature`
- `top_p` (not available for self-hosted models)
- `max_tokens`

Example of how to use the function:
```codeblock
from protollm.connectors.connector_creator import create_llm_connector

model = create_llm_connector("https://api.vsegpt.ru/v1;openai/gpt-4o-mini", temperature=0.015, top_p=0.95)
res = model.invoke("Tell me a joke")
print(res.content)
```
You can find the rest of the examples [here](https://github.com/ITMO-NSS-team/ProtoLLM/tree/main/examples/connector_creator_usage_examples.py)

## New connectors

If you want to add a new connector, you need to implement a class based on the BaseChatModel class with all the
necessary methods. Instructions for implementation available 
[here](https://python.langchain.com/docs/how_to/custom_chat_model/).


