
# LLM API Documentation

This API allows interaction with a distributed LLM architecture using RabbitMQ and Redis. Requests are processed asynchronously by a worker system (LLM-core) that generates responses and saves them to Redis. The API retrieves results from Redis and sends them back to the user.

---

## Endpoints

### `/generate`
- **Method**: `POST`
- **Description**: Sends a prompt for single message generation.
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "meta": {
      "temperature": 0.2,
      "tokens_limit": 8096,
      "stop_words": [
        "string"
      ],
      "model": "string"
    },
    "content": "string"
  }
  ```
  - `job_id` (string): Unique identifier for the task.
  - `meta` (object): Metadata for generation:
    - `temperature` (float): The degree of randomness in generation (default 0.2).
    - `tokens_limit` (integer): Maximum tokens for the response (default 8096).
    - `stop_words` (list of strings): Words to stop generation.
    - `model` (string): Model to use for generation.
  - `content` (string): The input text for generation.
- **Response**:
  ```json
  {
    "content": "string"
  }
  ```
  - `content` (string): The generated text.

---

### `/chat_completion`
- **Method**: `POST`
- **Description**: Sends a conversation history for chat-based completions.
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "meta": {
      "temperature": 0.2,
      "tokens_limit": 8096,
      "stop_words": [
        "string"
      ],
      "model": "string"
    },
    "messages": [
      {
        "role": "string",
        "content": "string"
      }
    ]
  }
  ```
  - `job_id` (string): Unique identifier for the task.
  - `meta` (object): Metadata for chat completion:
    - `temperature` (float): The degree of randomness in responses (default 0.2).
    - `tokens_limit` (integer): Maximum tokens for the response (default 8096).
    - `stop_words` (list of strings): Words to stop the generation.
    - `model` (string): Model to use for chat completion.
  - `messages` (list of objects): Conversation history:
    - `role` (string): Role of the message sender (`"user"`, `"assistant"`, etc.).
    - `content` (string): Message content.
- **Response**:
  ```json
  {
    "content": "string"
  }
  ```
  - `content` (string): The generated response.

---

## Environment Variables

These variables must be configured and synchronized with the LLM-core system:

### RabbitMQ Configuration
- `RABBIT_MQ_HOST`: RabbitMQ server hostname or IP.
- `RABBIT_MQ_PORT`: RabbitMQ server port.
- `RABBIT_MQ_LOGIN`: RabbitMQ login username.
- `RABBIT_MQ_PASSWORD`: RabbitMQ login password.
- `QUEUE_NAME`: Name of the RabbitMQ queue to process tasks.

### Redis Configuration
- `REDIS_HOST`: Redis server hostname or IP.
- `REDIS_PORT`: Redis server port.
- `REDIS_PREFIX`: Key prefix for task results in Redis.

### Internal LLM-core Configuration
- `INNER_LLM_URL`: URL for the LLM-core worker service.

### Example `.env` File
```env
# API
CELERY_BROKER_URL=amqp://admin:admin@127.0.0.1:5672/
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
REDIS_HOST=redis
REDIS_PORT=6379
RABBIT_MQ_HOST=rabbitmq
RABBIT_MQ_PORT=5672
RABBIT_MQ_LOGIN=admin
RABBIT_MQ_PASSWORD=admin
WEB_RABBIT_MQ=15672
API_PORT=6672

# RabbitMQ
RABBITMQ_DEFAULT_USER=admin
RABBITMQ_DEFAULT_PASS=admin
```

---

## System Architecture

Below is the architecture diagram for the interaction between API, RabbitMQ, LLM-core, and Redis:

```plaintext
+-------------------+       +-----------------+       +----------------+       +-------------------+
|                   |       |                 |       |                |       |                   |
|       API         +------>+    RabbitMQ     +------>+    LLM-core    +------>+      Redis         |
|                   |       |                 |       |                |       |                   |
+-------------------+       +-----------------+       +----------------+       +-------------------+
        ^                             ^                                ^
        |                             |                                |
        |      Requests are queued    |    Worker retrieves tasks     | Results are stored in Redis
        |      Results are polled     |                                |
        +-----------------------------+--------------------------------+
```

### Flow
1. **API**:
   - Receives requests via endpoints (`/generate`, `/chat_completion`).
   - Publishes tasks to RabbitMQ.
   - Polls Redis for results based on task IDs.

2. **RabbitMQ**:
   - Acts as a queue for task distribution.
   - LLM-core workers subscribe to queues to process tasks.

3. **LLM-core**:
   - Retrieves tasks from RabbitMQ.
   - Processes prompts or chat completions using LLM models.
   - Stores results in Redis.

4. **Redis**:
   - Acts as the result storage.
   - API retrieves results from Redis when tasks are completed.

---

## Usage

### Running the API
1. Configure environment variables in the `.env` file.
2. Start the API using:
```python
app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))
```
### Running the API Locally (without Docker)
To run the API locally using Uvicorn, use the following command:

```sh
uvicorn protollm_api.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Or use this main file:
```python
app = FastAPI()

config = Config.read_from_env()

app.include_router(get_router(config))
    
if __name__ == "__main__":
    uvicorn.run("protollm_api.backend.main:app", host="127.0.0.1", port=8000, reload=True)
```
### Example Request
#### Generate
```bash
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{
  "job_id": "12345",
  "meta": {
    "temperature": 0.5,
    "tokens_limit": 1000,
    "stop_words": ["stop"],
    "model": "gpt-model"
  },
  "content": "What is AI?"
}'
```

#### Chat Completion
```bash
curl -X POST "http://localhost:8000/chat_completion" -H "Content-Type: application/json" -d '{
  "job_id": "12345",
  "meta": {
    "temperature": 0.5,
    "tokens_limit": 1000,
    "stop_words": ["stop"],
    "model": "gpt-model"
  },
  "messages": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "Artificial Intelligence is..."}
  ]
}'
```

---

## Notes
- Ensure that `RABBIT_MQ_HOST`, `RABBIT_MQ_PORT`, `REDIS_HOST`, and other variables are synchronized between the API and LLM-core containers.
- The system supports distributed scaling by adding more LLM-core workers to the RabbitMQ queue.
