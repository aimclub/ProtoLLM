from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from protollm.connectors.rest_server import ChatRESTServer


def test_convert_messages_to_rest_server_messages():
    # Arrange
    msgs = [
        SystemMessage(content="s"),
        HumanMessage(content="h"),
        AIMessage(content="a"),
    ]

    # Act
    out = ChatRESTServer._convert_messages_to_rest_server_messages(msgs)

    # Assert
    assert out == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "h"},
        {"role": "assistant", "content": "a"},
    ]


def test_convert_messages_unsupported_type():
    # Arrange
    class X: ...

    msgs = [X()]  # type: ignore[list-item]

    # Act / Assert
    with pytest.raises(ValueError):
        ChatRESTServer._convert_messages_to_rest_server_messages(msgs)  # type: ignore[arg-type]


def test_create_chat_404():
    # Arrange
    conn = ChatRESTServer(base_url="http://x")
    with patch("requests.post") as post:
        post.return_value.status_code = 404
        post.return_value.text = "not found"

        # Act / Assert
        with pytest.raises(ValueError):
            conn._create_chat([HumanMessage(content="hi")])


def test_create_chat_200():
    # Arrange
    conn = ChatRESTServer(base_url="http://x")
    with patch("requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.text = '{"content": "ok"}'

        # Act
        res = conn._create_chat([HumanMessage(content="hi")])

    # Assert
    assert res["content"] == "ok"
