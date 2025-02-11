
# README.md

## Introduction

This repository provides a template for deploying large language models (LLMs) using Docker Compose. The setup is designed to integrate multiple models with GPU support, Redis for data storage, and RabbitMQ for task queuing and processing. The provided `main.py` script demonstrates how to initialize and run a connection to process tasks using any model that inherits from the base model class.

---

## Table of Contents

1. [Docker Compose Setup](#docker-compose-setup)
   - [Adding Multiple Models](#adding-multiple-models)
2. [Main Script Overview](#main-script-overview)
   - [Using Custom Models](#using-custom-models)
3. [Environment Variable Configuration](#environment-variable-configuration)
   - [Synchronization with API](#synchronization-with-api)

---

## Docker Compose Setup

The provided `docker-compose.yml` template can be used to deploy your LLM model(s). It supports GPU execution and integration with a shared network (`llm_wrap_network`).

### Example `docker-compose.yml`

```yaml
version: '3.8'

services:
   llm:
      container_name: <your_container_name>
      image: <your_image_name>:latest
      runtime: nvidia
      deploy:
         resources:
            limits:
               memory: 100G
      build:
         context: ..
         dockerfile: Dockerfile
      env_file: .env
      environment:
         FORCE_CMAKE: 1
      volumes:
         - <your_path_to_data_in_docker>:/data
      ports:
         - "8677:8672"
      networks:
         - llm_wrap_network
      restart: unless-stopped

networks:
   llm_wrap_network:
      name: llm_wrap_network
      driver: bridge
```

### Adding Multiple Models

You can define multiple models by duplicating the service block in `docker-compose.yml` and adjusting the relevant parameters (e.g., container name, ports, GPUs). For example:

```yaml
services:
  llm_1:
    container_name: llm_model_1
    image: llm_image:latest
    runtime: nvidia
    environment:
      MODEL_PATH: /data/model_1
      NVIDIA_VISIBLE_DEVICES: "GPU-1"
    ports:
      - "8677:8672"

  llm_2:
    container_name: llm_model_2
    image: llm_image:latest
    runtime: nvidia
    environment:
      MODEL_PATH: /data/model_2
      NVIDIA_VISIBLE_DEVICES: "GPU-2"
    ports:
      - "8678:8672"

networks:
  llm_wrap_network:
    name: llm_wrap_network
    driver: bridge
```

By assigning separate GPUs and ports, you can scale your infrastructure to serve multiple models simultaneously.

---

## Main Script Overview

The provided `main.py` script demonstrates how to initialize and run the LLM wrapper (`LLMWrap`) with a selected model. The wrapper uses RabbitMQ for task queuing and Redis for result storage.

### Key Components

1. **Model Initialization**:
   ```python
   llm_model = VllMModel(model_path=MODEL_PATH)
   ```
   - Any model inheriting from `BaseLLM` can be used here.
   - Replace `VllMModel` with your custom model class if needed.

2. **LLMWrap Initialization**:
   ```python
   llm_wrap = LLMWrap(
       llm_model=llm_model,
       redis_host=REDIS_HOST,
       redis_port=REDIS_PORT,
       queue_name=QUEUE_NAME,
       rabbit_host=RABBIT_MQ_HOST,
       rabbit_port=RABBIT_MQ_PORT,
       rabbit_login=RABBIT_MQ_LOGIN,
       rabbit_password=RABBIT_MQ_PASSWORD,
       redis_prefix=REDIS_PREFIX
   )
   ```
   - This connects the model to the task queue and result storage.
   - Ensure the environment variables match the corresponding API configuration.

3. **Starting the Connection**:
   ```python
   llm_wrap.start_connection()
   ```
   - Begins consuming tasks from RabbitMQ and processes them using the selected model.
   - Results are saved to Redis.

---

## Environment Variable Configuration

Environment variables are defined in the `.env` file and passed to Docker Compose. Below are the required variables:

```plaintext
TOKENS_LEN=16384
GPU_MEMORY_UTILISATION=0.9
TENSOR_PARALLEL_SIZE=2
MODEL_PATH=/data/<your_model_path>
NVIDIA_VISIBLE_DEVICES=<your_GPUs>
REDIS_HOST=localhost
REDIS_PORT=6379
QUEUE_NAME=<your_queue_name>
RABBIT_MQ_HOST=<rabbitmq_host>
RABBIT_MQ_PORT=<rabbitmq_port>
RABBIT_MQ_LOGIN=<rabbitmq_login>
RABBIT_MQ_PASSWORD=<rabbitmq_password>
REDIS_PREFIX=<redis_key_prefix>
```

### Synchronization with API

If you are deploying an API in another container, ensure the following:
1. **Environment Variables**:
   - Match the Redis and RabbitMQ configuration between the API and the LLM containers (e.g., `REDIS_HOST`, `RABBIT_MQ_HOST`).

2. **Network**:
   - Both containers must be on the same Docker network (`llm_wrap_network` in this example).

3. **Shared Queues**:
   - The `QUEUE_NAME` variable should be consistent across containers to ensure tasks are properly routed.

---

## Running the System

1. **Setup Docker Compose**:
   - Adjust `docker-compose.yml` and `.env` with your specific configuration.
   - Start the system:
     ```bash
     docker-compose up -d
     ```

2. **Verify Running Containers**:
   - Check active containers:
     ```bash
     docker ps
     ```

3. **Monitor Logs**:
   - To view logs for a specific container:
     ```bash
     docker logs -f <container_name>
     ```

4. **Submit Tasks**:
   - Tasks can be submitted to the RabbitMQ queue, and results will be saved in Redis.

---

For more details, refer to the comments in `docker-compose.yml` and `main.py`. If you encounter any issues, feel free to open an issue in the repository!
