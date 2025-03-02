import pytest

from unittest.mock import MagicMock, AsyncMock, patch
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult

from protollm_sdk.object_interface.mongo_db_wrapper import MongoDBWrapper

@pytest.fixture
def wrapper_fixture():
    return MongoDBWrapper(
        mongodb_host="localhost",
        mongodb_port="27017",
        mongodb_user="admin",
        mongodb_password="admin",
        database="unittest_database",
        collection="unittest_collection"
    )

@patch.object(MongoDBWrapper, attribute="get_mongo_client")
def test_insert_single_document(mock_client, wrapper_fixture):
    """Tests the synchronous document insert performance correction"""
    mock_collection = MagicMock()

    # since get_mongo_client() is a context manager, it's __enter__ magic method need to be modified by mock_collection
    mock_client.return_value.__enter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }
    mock_collection.insert_one.return_value = InsertOneResult(inserted_id="test_id", acknowledged=True)

    result = wrapper_fixture.insert_single_document({"k": "v"})

    assert result.acknowledged is True
    assert result.inserted_id == "test_id"

@pytest.mark.asyncio
@patch.object(MongoDBWrapper, attribute="async_get_mongo_client")
async def test_async_insert_single_document(mock_client, wrapper_fixture):
    """Tests the asynchronous document insert performance correction"""
    mock_collection = AsyncMock()

    # since async_get_mongo_client() is a context manager, it's __aenter__ magic method need to be modified by mock_collection
    mock_client.return_value.__aenter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }

    mock_collection.insert_one.return_value = InsertOneResult(inserted_id="async_test_id", acknowledged=True)

    result = await wrapper_fixture.async_insert_single_document({"async_k": "async_v"})

    assert result.acknowledged is True
    assert result.inserted_id == "async_test_id"

@pytest.mark.asyncio
@patch.object(MongoDBWrapper, attribute="async_get_mongo_client")
async def test_insert_many_documents(mock_client, wrapper_fixture):
    """Tests the correction of asynchronous insertion of a set of documents"""
    mock_collection = AsyncMock()

    # since async_get_mongo_client() is a context manager, it's __aenter__ magic method need to be modified by mock_collection
    mock_client.return_value.__aenter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }
    mock_collection.insert_many.return_value = InsertManyResult(inserted_ids=["test_id1", "test_id2"], acknowledged=True)

    result = await wrapper_fixture.insert_many_documents(documents=[{"k1": "v1"}, {"k2": "v2"}])

    assert result.acknowledged is True
    assert len(result.inserted_ids) == 2

@pytest.mark.asyncio
@patch.object(MongoDBWrapper, attribute="async_get_mongo_client")
async def test_update_single_document(mock_client, wrapper_fixture):
    """Tests the correction of asynchronous document modification"""
    mock_collection = AsyncMock()

    # since async_get_mongo_client() is a context manager, it's __aenter__ magic method need to be modified by mock_collection
    mock_client.return_value.__aenter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }
    mock_collection.update_one.return_value = UpdateResult(raw_result={"nModified": 1}, acknowledged=True)

    result = await wrapper_fixture.update_single_document(pattern={"k": "v"}, update={"field": "content"})

    assert result.acknowledged is True
    assert result.raw_result["nModified"] == 1

@pytest.mark.asyncio
@patch.object(MongoDBWrapper, attribute="async_get_mongo_client")
async def test_get_single_document(mock_client, wrapper_fixture):
    """Tests the correction of asynchronous document fetching"""
    mock_collection = AsyncMock()

    # since async_get_mongo_client() is a context manager, it's __aenter__ magic method need to be modified by mock_collection
    mock_client.return_value.__aenter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }
    mock_collection.find_one.return_value = {"k": "v"}

    result = await wrapper_fixture.get_single_document(pattern={"k": "v"})

    assert result["k"] == "v"

@pytest.mark.asyncio
@patch.object(MongoDBWrapper, attribute="async_get_mongo_client")
async def test_delete_single_document(mock_client, wrapper_fixture):
    """Tests the correction of asynchronous document deletion"""
    mock_collection = AsyncMock()

    # since async_get_mongo_client() is a context manager, it's __aenter__ magic method need to be modified by mock_collection
    mock_client.return_value.__aenter__.return_value = {
        wrapper_fixture.database: {wrapper_fixture.collection: mock_collection}
    }
    mock_collection.delete_one.return_value = DeleteResult(raw_result={"n": 1}, acknowledged=True)

    result = await wrapper_fixture.delete_single_document({"k": "v"})

    assert result.acknowledged is True
    assert result.raw_result["n"] == 1
