from urllib.parse import urljoin

import httpx


class VectorDB:
    """
    VectorDB client
    """

    def __init__(self, vector_bd_host: str, vector_db_port: str | int | None = None):
        """
        Initialize VectorDB

        :param vector_bd_host: host of vector db
        :type vector_bd_host: str
        :param vector_db_port: port of vector db
        :type vector_db_port: str | int | None
        """
        self.url = (
            f"http://{vector_bd_host}:{vector_db_port}"
            if vector_db_port is not None
            else f"http://{vector_bd_host}"
        )
        self.client = httpx.Client()

    def api_v1(self):
        """
        Get api v1

        :return: response
        """
        response = self.client.get(
            urljoin(self.url, "/api/v1"),
            headers={"Content-type": "application/json"},
            timeout=10 * 60,
        )
        return response.json()
