import uuid
import pytest


def test_get_agents(docker_services, test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for agent in response.json():
        assert agent.get("name") in [
            "Ensemble",
            "Router",
            "rag_environment",
            "rag_education",
            "rag_union",
        ]
        assert "description" in agent
        assert "arguments" in agent
        assert isinstance(agent.get("arguments"), dict)
    assert len(response.json()) == 5


@pytest.mark.parametrize(
    "agent_id, agent_name",
    [
        ("07dd7db1-075a-4391-b537-6fbca4d5a5f6", "rag_environment"),
        ("3208a446-d847-45a8-a724-159fa87334b9", "rag_education"),
        ("2fb8e8f0-bd05-5eca-8e4d-376ede293e52", "rag_union"),
    ],
)
def test_get_agent_by_id(docker_services, test_client, agent_id, agent_name):
    response = test_client.get(f"/{agent_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert response.json().get("name") == agent_name


def test_get_agent_by_id_not_found(docker_services, test_client):
    response = test_client.get(f"/{uuid.uuid4()}")
    assert response.status_code == 404


def test_ensemble_websocket(docker_services, test_client, create_collections):
    with test_client.websocket_connect("/ensemble") as websocket:
        json_data = {
            "dialogue_id": str(uuid.uuid4()),
            "chat_history": [],
            "query": "Какие целевые показатели госпрограмм по образованию и защите окружающей среды?",
        }
        websocket.send_json(json_data)
        is_eos = False
        events = set()
        while not is_eos:
            data = websocket.receive_json()
            print(data.get("event_id"))
            event_name = data.get("name")
            assert event_name != "error"
            is_eos = data.get("is_eos", False)
            events.add(event_name)
        assert events == set({"answer", "tool_answer", "retrieval"})


def test_router_websocket(docker_services, test_client, create_collections):
    with test_client.websocket_connect("/router") as websocket:
        json_data = {
            "dialogue_id": str(uuid.uuid4()),
            "chat_history": [],
            "query": "Какие целевые показатели госпрограмм по образованию?",
        }
        websocket.send_json(json_data)
        is_eos = False
        events = set()
        while not is_eos:
            data = websocket.receive_json()
            event_name = data.get("name")
            assert event_name != "error"
            is_eos = data.get("is_eos", False)
            events.add(event_name)
        assert events == set({"answer", "tool_answer", "retrieval"})


@pytest.mark.parametrize(
    "agent_id",
    [
        "07dd7db1-075a-4391-b537-6fbca4d5a5f6",
        "3208a446-d847-45a8-a724-159fa87334b9",
        "2fb8e8f0-bd05-5eca-8e4d-376ede293e52",
    ],
)
def test_rag_websocket(docker_services, test_client, agent_id, create_collections):
    with test_client.websocket_connect("/agent") as websocket:
        json_data = {
            "dialogue_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "chat_history": [],
            "query": "Какие целевые показатели госпрограмм по образованию и защите окружающей среды?",
            "run_params": {},
        }
        websocket.send_json(json_data)
        is_eos = False
        events = set()
        while not is_eos:
            data = websocket.receive_json()
            event_name = data.get("name")
            assert event_name != "error"
            is_eos = data.get("is_eos", False)
            events.add(event_name)
        assert events == set({"answer", "retrieval"})
