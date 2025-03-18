import pytest

from protollm_api.config import Config


@pytest.fixture(scope="module")
def test_local_config():
    return Config()


@pytest.fixture(scope="module")
def test_real_config():
    return Config.read_from_env()
