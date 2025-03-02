import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from pymongo import MongoClient
from pymongo.results import InsertOneResult, InsertManyResult, DeleteResult, UpdateResult

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

class MongoDBWrapper:
    """Implementation of the interface used for interaction with MongoDB"""

    def __init__(
            self,
            mongodb_host: str,
            mongodb_port: int | str,
            mongodb_user: str,
            mongodb_password: str,
            database: str,
            collection: str,
            **connection_args
    ) -> None:
        self.conn = f"mongodb://{mongodb_user}:{mongodb_password}@{mongodb_host}:{mongodb_port}"
        self.database = database
        self.collection = collection

        # you could try defaults like this: {"zlibCompressionLevel": 7, "compressors": "zlib"}
        self.connection_args = connection_args


    @asynccontextmanager
    async def async_get_mongo_client(self) -> AsyncIOMotorClient:
        """Async context manager for providing access to the async MongoDB client."""
        client = AsyncIOMotorClient(self.conn, **self.connection_args)

        try:
            yield client
        finally:
            client.close()

    @contextmanager
    def get_mongo_client(self) -> MongoClient:
        """Context manager for providing access to the MongoDB client."""
        client = MongoClient(self.conn, **self.connection_args)

        try:
            yield client
        finally:
            client.close()

    def insert_single_document(self, document: dict[Any, Any]) -> InsertOneResult:
        """
        Inserts a single transmitted document into MongoDB collection.
        Async version ot this method is available.

        Args:
            document (dict[Any, Any]): content to insert in MongoDB format.

        Returns:
            Inserted document identifier according to InsertOneResult class.
        """
        try:
            with self.get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: MongoClient = client[self.database][self.collection]

                return _cursor.insert_one(document)

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex

    async def async_insert_single_document(self, document: dict[Any, Any]) -> InsertOneResult:
        """
        Inserts a single transmitted document into MongoDB collection.

        Args:
            document (dict[Any, Any]): content to insert in MongoDB format.

        Returns:
            Inserted document identifier according to InsertOneResult class.
        """
        try:
            async with self.async_get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: AsyncIOMotorCollection = client[self.database][self.collection]

                return await _cursor.insert_one(document)

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex

    async def insert_many_documents(self, documents: list[dict[Any, Any]]) -> InsertManyResult:
        """
        Inserts a several transmitted documents into MongoDB collection.

        Args:
            documents (list[dict[Any, Any]]): content to insert in MongoDB format.

        Returns:
            Inserted documents identifiers according to InsertManyResult class.

        Raises:
            Exceptions that might happen throughout connection attempt or performing insert statement.
        """
        try:
            async with self.async_get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: AsyncIOMotorCollection = client[self.database][self.collection]

                return await _cursor.insert_many(documents)

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex

    async def get_single_document(self, pattern: dict) -> dict:
        """
        Fetches the document from MongoDB collection.

        Args:
            pattern (dict): the pattern under which the document will be searched in MongoDB.

        Returns:
            The first fetched document for transmitted search pattern

        Raises:
            Exceptions that might happen throughout connection attempt or fetching documents.
        """
        try:
            async with self.async_get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: AsyncIOMotorCollection = client[self.database][self.collection]
                return await _cursor.find_one(pattern)

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex

    async def update_single_document(self, pattern: dict[Any, Any], update: dict[Any, Any]) -> UpdateResult:
        """
        Performs modification of the document by transmitted content.

        Args:
            pattern dict[Any, Any]: the pattern under which the document will be searched in MongoDB
            update dict[Any, Any]: update operation content

        Returns:

        """
        try:
            async with self.async_get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: AsyncIOMotorCollection = client[self.database][self.collection]

                return await _cursor.update_one(pattern, update={"$set": update})

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex

    async def delete_single_document(self, pattern: dict[Any, Any]) -> DeleteResult:
        """
        Deletes a document by transmitted searching pattern in MongoDB collection.

        Args:
            pattern dict[Any, Any]: the pattern under which the document will be searched and deleted in MongoDB.

        Returns:
            operation result according to DeleteResult class

        Raises:
            Exceptions that might happen throughout connection attempt or performing delete statement.
        """
        try:
            async with self.async_get_mongo_client() as client:

                # _cursor might be called as collection
                _cursor: AsyncIOMotorCollection = client[self.database][self.collection]

                return await _cursor.delete_one(pattern)

        except Exception as ex:
            logger.info(f"{ex}")
            raise ex
