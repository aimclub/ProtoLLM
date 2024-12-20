import logging

from protollm_agents.entrypoint import Entrypoint


logging.basicConfig(
level=logging.DEBUG,
format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    epoint = Entrypoint(config_path="./examples/admin-config.yml")
    epoint.run()
