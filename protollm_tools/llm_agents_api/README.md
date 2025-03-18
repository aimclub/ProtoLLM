# llm-agents-api

This tool library provides a simple API for creating and running LLM agents and build multi-agent systems.
SDK allows agents creation and management, and provides interface to integrate those using router agent and ensemble agent.
Tool also provides an Entrypoint object which starts uvicorn server for running the API.

## Installation

```bash
poetry install
```

## Run example
1) Copy .env.example to .env and set variables
2) Run
```bash
docker compose up -d
```
3) Run example
```bash
poetry run python examples/main.py
```

4) Open browser and go to http://<app_host>:<app_port>/docs (by default http://0.0.0.0:8080/docs):
- `/` - Agents listing
- `/agents/<idx>` - Agent details

5) You can use any Websocket client to connect to the agents and send messages. For example, you can use Postman.
- `ws://<app_host>:<app_port>/agents` - Websocket connection to the agent
Example query:
```json
{
    "dialogue_id": "2fb8e8f0-bd05-5eca-8e4d-376ede293e53",
    "agent_id": "3208a446-d847-45a8-a724-159fa87334b9",
    "chat_history":[],
    "query":"Какие целевые показатели госпрограмм по образованию и защите окружающей среды?",
    "run_params": {}
}
```
- `ws://<app_host>:<app_port>/router` - Websocket connection to the router agent
Example query:
```json
{
    "dialogue_id": "2fb8e8f0-bd05-5eca-8e4d-376ede293e53",
    "chat_history":[],
    "query":"Какие целевые показатели госпрограмм по образованию и защите окружающей среды?"
}
```
- `ws://<app_host>:<app_port>/ensemble` - Websocket connection to the ensemble agent
Example query:
```json
{
    "dialogue_id": "2fb8e8f0-bd05-5eca-8e4d-376ede293e53",
    "chat_history":[],
    "query":"Какие целевые показатели госпрограмм по образованию и защите окружающей среды?"
}
```

After the message is sent, you will receive a stream of messages from the agent. 

## Run tests
1) Copy .env.test.example to .env.test and set variables
2) Run
```bash
poetry run pytest
```
