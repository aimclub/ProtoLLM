import os


class Config:
    """
       Configuration class for setting up Redis, RabbitMQ, and model-specific parameters.

       Attributes:
           redis_host: The hostname of the Redis server. Defaults to "localhost".
           redis_port: The port number of the Redis server. Defaults to 6379.
           redis_prefix: Prefix for keys used in Redis. Defaults to "llm-api".
           rabbit_host: The hostname of the RabbitMQ server. Defaults to "localhost".
           rabbit_port: The port number of the RabbitMQ server. Defaults to 5672.
           rabbit_login: The username for RabbitMQ authentication. Defaults to "admin".
           rabbit_password: The password for RabbitMQ authentication. Defaults to "admin".
           queue_name: The name of the RabbitMQ queue to use. Defaults to "llm-api-queue".
           mongodb_host: The hostname of the MongoDB server. Defaults to "localhost".
           mongodb_port: The port number of the MongoDB server. Defaults to 27017.
           mongodb_user: The username for MongoDB authentication. Defaults to "admin".
           mongodb_password: The password for MongoDB authentication. Defaults to "admin".
           mongodb_database_name: The name of the working database in MongoDB. Defaults to "llm-api-database".
           mongodb_collection_name: The name of the working collection inside database in MongoDB. Defaults to "llm-api-collection".
           model_path: Path to the model being used. Defaults to None.
           token_len: The maximum length of tokens for processing by the model. Defaults to None.
           tensor_parallel_size: The size of tensor parallelism for distributed processing. Defaults to None.
           gpu_memory_utilisation: The percentage of GPU memory utilization for the model. Defaults to None.
    """

    def __init__(
            self,
            redis_host: str = "localhost",
            redis_port: int = 6379,
            redis_prefix: str = "llm-api",
            rabbit_host: str = "localhost",
            rabbit_port: int = 5672,
            rabbit_login: str = "admin",
            rabbit_password: str = "admin",
            queue_name: str = "llm-api-queue",
            mongodb_host: str = "localhost",
            mongodb_port: int = 27017,
            mongodb_user: str = "admin",
            mongodb_password: str = "admin",
            mongodb_database_name: str = "llm-api-database",
            mongodb_collection_name: str = "llm-api-collection",
            model_path: str = None,
            token_len: int = None,
            tensor_parallel_size: int = None,
            gpu_memory_utilisation: float = None,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_prefix = redis_prefix
        self.rabbit_host = rabbit_host
        self.rabbit_port = rabbit_port
        self.rabbit_login = rabbit_login
        self.rabbit_password = rabbit_password
        self.queue_name = queue_name
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.mongodb_user = mongodb_user
        self.mongodb_password = mongodb_password
        self.mongodb_database_name = mongodb_database_name
        self.mongodb_collection_name = mongodb_collection_name
        self.model_path = model_path,
        self.token_len = token_len,
        self.tensor_parallel_size = tensor_parallel_size,
        self.gpu_memory_utilisation = gpu_memory_utilisation,

    @classmethod
    def read_from_env(cls) -> 'Config':
        return Config(
            os.environ.get("REDIS_HOST"),
            int(os.environ.get("REDIS_PORT")),
            os.environ.get("REDIS_PREFIX"),
            os.environ.get("RABBIT_MQ_HOST"),
            int(os.environ.get("RABBIT_MQ_PORT")),
            os.environ.get("RABBIT_MQ_LOGIN"),
            os.environ.get("RABBIT_MQ_PASSWORD"),
            os.environ.get("QUEUE_NAME"),
            int(os.environ.get("MONGO_DB_PORT")),
            os.environ.get("MONGO_DB_HOST"),
            os.environ.get("MONGODB_USER"),
            os.environ.get("MONGODB_PASS"),
            int(os.environ.get("WEB_MONGO_DB_PORT")),
            os.environ.get("MODEL_PATH"),
            int(os.environ.get("TOKENS_LEN")),
            int(os.environ.get("TENSOR_PARALLEL_SIZE")),
            float(os.environ.get("GPU_MEMORY_UTILISATION")),
        )

    @classmethod
    def read_from_env_file(cls, path: str) -> 'Config':
        with open(path) as file:
            lines = file.readlines()
        env_vars = {}
        for line in lines:
            key, value = line.split("=")
            env_vars[key] = value
        return Config(
            env_vars.get("REDIS_HOST"),
            int(env_vars.get("REDIS_PORT")),
            env_vars.get("REDIS_PREFIX"),
            env_vars.get("RABBIT_MQ_HOST"),
            int(env_vars.get("RABBIT_MQ_PORT")),
            env_vars.get("RABBIT_MQ_LOGIN"),
            env_vars.get("RABBIT_MQ_PASSWORD"),
            env_vars.get("QUEUE_NAME"),
            int(env_vars.get("MONGO_DB_PORT")),
            env_vars.get("MONGO_DB_HOST"),
            env_vars.get("MONGODB_USER"),
            env_vars.get("MONGODB_PASS"),
            int(env_vars.get("WEB_MONGO_DB_PORT")),
            env_vars.get("MODEL_PATH"),
            int(env_vars.get("TOKENS_LEN")),
            int(env_vars.get("TENSOR_PARALLEL_SIZE")),
            float(env_vars.get("GPU_MEMORY_UTILISATION")),
        )