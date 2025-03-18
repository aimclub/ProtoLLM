from enum import Enum
import uuid
import requests
from websockets import ConnectionClosed
import websockets.sync.client as wsclient
import json
import click
import pandas as pd


class AnswerType(str, Enum):
    RETRIEVAL = "retrieval"
    ANSWER = "answer"
    ERROR = "error"


def parse_ws_response(response):
    response_body = json.loads(response)
    match (name := response_body.get("name")):
        case AnswerType.RETRIEVAL | AnswerType.ANSWER:
            return name, response_body["result"]
        case AnswerType.ERROR:
            raise Exception(response_body.get("detail"))
        case _:
            return None, None


@click.command()
@click.option("--basepath", type=str, default="0.0.0.0:8080")
@click.option("--output", type=str, default="agents_responses.csv")
def main(basepath, output):
    run_uid = str(uuid.uuid4())
    questions = [
        "Какой объем финансирования программы политики защиты окружающей среды",
        "Какая разница в объеме финансирования программ защиты окружающей среды и образования?",
        "Кто ответственный исполнитель программы политики защиты окружающей среды?",
        "Какие целевые показатели госпрограмм по образованию и защите окружающей среды?",
        "Сколько подпрограмм в госполитике по защите окружающей среды?",
        "Какие приоритеты политики в плане обращений с твердыми коммунальными отходами?",
        "какой объем финансирования программы образования в 2017 году?",
    ]

    response = requests.get(f"http://{basepath}/", params={"agent_type": "streaming"})
    assert response.status_code == 200, "Failed to get agents"
    response = response.json()
    assert len(response) > 0, "No agents found"
    agents_ids = {
        agent["agent_id"]: agent["name"] for agent in response if "rag" in agent["name"]
    }

    agents_responses = list()
    for question in questions:
        question_columns = dict(question=question)
        for agent_id, agent_name in agents_ids.items():
            click.echo(f"Collecting response from {agent_name}, {question=}")
            with wsclient.connect(f"ws://{basepath}/agent") as ws:
                request_payload = {
                    "dialogue_id": run_uid,
                    "agent_id": agent_id,
                    "chat_history": [],
                    "query": question,
                    "run_params": {},
                }
                ws.send(json.dumps(request_payload))
                try:
                    while True:
                        response = ws.recv()
                        response_type, response_data = parse_ws_response(response)
                        if response_type == AnswerType.RETRIEVAL:
                            question_columns[f"docs_{agent_name}"] = response_data
                        elif response_type == AnswerType.ANSWER:
                            question_columns[f"answer_{agent_name}"] = response_data
                except ConnectionClosed:
                    pass
        click.echo("Finished collecting RAG agents responses")

        for endpoint in ("router", "ensemble"):
            click.echo(f"Collecting response from {endpoint}, {question=}")
            with wsclient.connect(f"ws://{basepath}/{endpoint}") as ws:
                request_payload = {
                    "dialogue_id": run_uid,
                    "chat_history": [],
                    "query": question,
                }
                ws.send(json.dumps(request_payload))
                try:
                    while True:
                        response = ws.recv()
                        response_type, response_data = parse_ws_response(response)
                        if response_type == AnswerType.RETRIEVAL:
                            question_columns[f"docs_{endpoint}"] = response_data
                        elif response_type == AnswerType.ANSWER:
                            question_columns[f"answer_{endpoint}"] = response_data
                except ConnectionClosed:
                    pass
        click.echo("Finished collecting router and ensemble responses")
        click.echo(f"Collected question_columns: {question_columns}")
        agents_responses.append(question_columns)
    click.echo("Finished collecting agents responses")

    df = pd.DataFrame().from_records(agents_responses)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
