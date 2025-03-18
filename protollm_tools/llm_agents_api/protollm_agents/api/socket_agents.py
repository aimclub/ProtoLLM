import logging
import uuid

from pydantic import ValidationError
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from protollm_agents.services.agents_manager import AgentsManager, get_agents_manager

from protollm_agents.models.requests import AgentSocketQuery, RouterSocketQuery
from protollm_agents.models.responses import ErrorMessage
from protollm_agents.services.exceptions import AgentNotFound
from protollm_agents.services.socket_connector import (
    get_socket_connector,
    SocketConnector,
)
from protollm_agents.services.storage import Storage, get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/agent")
async def agent_call(
    websocket: WebSocket,
    connector: SocketConnector = Depends(get_socket_connector),
    manager: AgentsManager = Depends(get_agents_manager),
    storage: Storage = Depends(get_storage),
):
    await connector.connect(websocket)
    try:
        unique_query_uuid = str(uuid.uuid4())
        logger.info(f"{unique_query_uuid} - Connected")
        data = await websocket.receive_json()
        query_data = AgentSocketQuery.model_validate(data)
        async for event in manager.run_streaming_agent(
            agent_id=query_data.agent_id,
            arguments=query_data.run_params,
            history=query_data.history_as_tuple_list,
            query=query_data.query,
            storage=storage,
        ):
            await connector.send_encoded_model(
                manager.prepare_event(agent_id=event.agent_id, event=event)
            )
        await websocket.close()
    except (ValidationError, AgentNotFound) as exc:
        await connector.send_encoded_model(ErrorMessage(detail=str(exc)))
        await websocket.close()
        raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(str(exc))
        await websocket.close()
        raise exc
    finally:
        await connector.disconnect()


@router.websocket("/router")
async def router_call(
    websocket: WebSocket,
    connector: SocketConnector = Depends(get_socket_connector),
    manager: AgentsManager = Depends(get_agents_manager),
    storage: Storage = Depends(get_storage),
):
    await connector.connect(websocket)
    try:
        unique_query_uuid = str(uuid.uuid4())
        logger.info(f"{unique_query_uuid} - Connected")
        data = await websocket.receive_json()
        query_data = RouterSocketQuery.model_validate(data)
        async for event in manager.run_router_agent(
            history=query_data.history_as_tuple_list,
            query=query_data.query,
            storage=storage,
        ):
            await connector.send_encoded_model(
                manager.prepare_event(agent_id=event.agent_id, event=event)
            )
        await websocket.close()
    except (ValidationError, AgentNotFound) as exc:
        await connector.send_encoded_model(ErrorMessage(detail=str(exc)))
        await websocket.close()
        raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(str(exc))
        await websocket.close()
        raise exc
    finally:
        await connector.disconnect()


@router.websocket("/ensemble")
async def ensemble_call(
    websocket: WebSocket,
    connector: SocketConnector = Depends(get_socket_connector),
    manager: AgentsManager = Depends(get_agents_manager),
    storage: Storage = Depends(get_storage),
):
    await connector.connect(websocket)
    try:
        unique_query_uuid = str(uuid.uuid4())
        logger.info(f"{unique_query_uuid} - Connected")
        data = await websocket.receive_json()
        query_data = RouterSocketQuery.model_validate(data)
        async for event in manager.run_ensemble_agent(
            history=query_data.history_as_tuple_list,
            query=query_data.query,
            storage=storage,
        ):
            await connector.send_encoded_model(
                manager.prepare_event(agent_id=event.agent_id, event=event)
            )
        await websocket.close()
    except (ValidationError, AgentNotFound) as exc:
        await connector.send_encoded_model(ErrorMessage(detail=str(exc)))
        await websocket.close()
        raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(str(exc))
        await websocket.close()
        raise exc
    finally:
        await connector.disconnect()
