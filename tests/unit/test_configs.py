import importlib.util
from pathlib import Path

import pytest


def _load_entrypoint_config_class():
    root = Path(__file__).resolve().parents[2]
    cfg_path = (
        root / "protollm_tools" / "llm-agents-api" / "protollm_agents" / "configs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "protollm_tools.llm-agents-api.protollm_agents.configs", str(cfg_path)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, "EntrypointConfig")


def test_entrypoint_config_all_none_is_not_admin():
    # Arrange & Act
    EntrypointConfig = _load_entrypoint_config_class()
    cfg = EntrypointConfig()

    # Assert
    assert cfg.is_admin is False


def test_entrypoint_config_all_provided_is_admin():
    # Arrange & Act
    EntrypointConfig = _load_entrypoint_config_class()
    cfg = EntrypointConfig(
        redis_host="h",
        redis_port=6379,
        redis_db=0,
        postgres_host="h",
        postgres_port=5432,
        postgres_user="u",
        postgres_password="p",
        postgres_db="db",
    )

    # Assert
    assert cfg.is_admin is True


@pytest.mark.parametrize(
    "partial",
    [
        dict(redis_host="h"),
        dict(redis_host="h", redis_port=6379),
        dict(postgres_host="h"),
        dict(postgres_host="h", postgres_port=5432, postgres_user="u"),
    ],
)
def test_entrypoint_config_partial_params_raises(partial):
    # Arrange / Act / Assert
    EntrypointConfig = _load_entrypoint_config_class()
    with pytest.raises(ValueError):
        EntrypointConfig(**partial)
