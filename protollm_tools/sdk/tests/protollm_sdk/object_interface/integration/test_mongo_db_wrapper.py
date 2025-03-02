import pytest

from pymongo.collection import Collection
from protollm_sdk.object_interface.mongo_db_wrapper import MongoDBWrapper

@pytest.fixture(scope="module")
def wrapper_fixture():
    return MongoDBWrapper(
        mongodb_host="localhost",
        mongodb_port="27017",
        mongodb_user="admin",
        mongodb_password="admin",
        database="unittest_database",
        collection="unittest_collection"
    )

@pytest.fixture(autouse=True)
def drop_collection_fixture(wrapper_fixture):
    """Drops test collection before each test case"""
    with wrapper_fixture.get_mongo_client() as client:
        _cursor: Collection = client[wrapper_fixture.database][wrapper_fixture.collection]
        return _cursor.drop()


@pytest.mark.local
def test_insert_single_document(wrapper_fixture):
    """Tests the synchronous document insert performance correction"""

    insert_result = wrapper_fixture.insert_single_document(document={"_id": "test_id", "content": "test_content"})

    assert insert_result.acknowledged is True
    assert insert_result.inserted_id == "test_id"

    with wrapper_fixture.get_mongo_client() as client:
        _cursor: Collection = client[wrapper_fixture.database][wrapper_fixture.collection]

        document = _cursor.find_one(filter={"_id": "test_id"})
        assert document["content"] == "test_content"


@pytest.mark.asyncio
async def test_async_insert_single_document(wrapper_fixture):
    """Tests the asynchronous document insert performance correction"""

    asynct_insert_result = await wrapper_fixture.async_insert_single_document(
        document={"_id": "async_test_id", "content": "async_test_content"}
    )

    assert asynct_insert_result.acknowledged is True
    assert asynct_insert_result.inserted_id == "async_test_id"

    async with wrapper_fixture.async_get_mongo_client() as client:
        _cursor: Collection = client[wrapper_fixture.database][wrapper_fixture.collection]

        fetch_result = await _cursor.find_one(filter={"_id": "async_test_id"})
        assert fetch_result["content"] == "async_test_content"

@pytest.mark.asyncio
async def test_insert_many_documents(wrapper_fixture):
    """Tests the correction of asynchronous insertion of a set of documents"""

    insert_many_result = await wrapper_fixture.insert_many_documents(
        documents=[{"_id": "k1", "content": "v1"}, {"_id": "k2", "content": "v2"}]
    )
    assert insert_many_result.acknowledged is True
    assert len(insert_many_result.inserted_ids) == 2

@pytest.mark.asyncio
async def test_update_single_document(wrapper_fixture):
    """Tests the correction of asynchronous document modification"""

    # inserting a test document which should be modified
    await wrapper_fixture.async_insert_single_document(document={"_id": "k1"})

    update_result = await wrapper_fixture.update_single_document(pattern={"_id": "k1"}, update={"content": "update"})
    assert update_result.acknowledged is True
    assert update_result.raw_result["nModified"] == 1

    # validation of success document modification
    async with wrapper_fixture.async_get_mongo_client() as client:
        _cursor: Collection = client[wrapper_fixture.database][wrapper_fixture.collection]

        fetch_result = await _cursor.find_one(filter={"_id": "k1"})
        assert fetch_result["content"] == "update"

@pytest.mark.asyncio
async def test_get_single_document(wrapper_fixture):
    """Tests the correction of asynchronous document fetching"""

    # inserting a test document which should be extracted from collection
    await wrapper_fixture.async_insert_single_document(document={"_id": "k1", "content": "test_content"})

    fetch_result = await wrapper_fixture.get_single_document(pattern={"_id": "k1"})
    assert (fetch_result.get("_id") == "k1") & (fetch_result.get("content") == "test_content")

@pytest.mark.asyncio
async def test_delete_single_document(wrapper_fixture):
    """Tests the correction of asynchronous document deletion"""

    # inserting a test document which should be extracted from collection
    await wrapper_fixture.async_insert_single_document(document={"_id": "k1", "content": "test_content"})

    delete_result = await wrapper_fixture.delete_single_document(pattern={"_id": "k1"})
    assert delete_result.acknowledged is True
    assert delete_result.raw_result["n"] == 1
