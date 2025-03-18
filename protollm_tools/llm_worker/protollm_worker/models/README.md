
# Introduction to Language Models in Your Project

This repository contains implementations of language models for various tasks, 
such as single-prompt generation and chat-based completions. The models are built 
on top of both local and API-based frameworks, providing flexibility and scalability.

## Table of Contents
1. [Overview](#overview)
2. [Models](#models)
   - [BaseLLM](#basellm)
   - [LocalLLM](#localllm)
     - [vLLM Model](#vllm-model)
     - [cpp Model](#cpp-model)
     - [hf Model](#hf-model)
   - [API-based Models](#api-based-models)
     - [OpenAPILLM](#openapillm)


---

## Overview

The project provides a unified interface for working with various types of language models, including:
- **Local models**: Models that are stored and executed on the local system.
- **API-based models**: Models that interact with external APIs (e.g., OpenAI API).


Each model is designed to handle prompt-based and chat-based transactions with customizable parameters like token limits, temperature, and stop words.

---

## Models

### BaseLLM
`BaseLLM` serves as an abstract base class for all language model implementations. It defines the basic structure and methods that other models inherit and implement.

**Key Features**:
- Abstract methods for generating text and handling completions.
- Ensures consistency across different implementations.

---

### LocalLLM
`LocalLLM` is designed for language models stored locally on your system. It leverages a local file path to initialize the model.

**Methods**:
- `generate`: Generates text based on a single prompt.
- `create_completion`: Handles chat-like interactions using a series of messages.

---

### vLLM Model
`VllMModel` uses the vLLM framework to enable high-performance local model execution. It supports both single-prompt and chat-based completions.

**Example Configuration**:
```python
model = VllMModel(model_path="/path/to/your/model")
```

---

### cpp Model
`cpp_Model` in development

---

### hf Model
`VllMModel` in development

---

### API-based Models

`API-based LLM` interacts with external APIs, such as OpenAI, to generate responses. It uses the OpenAI Python SDK for communication.

**Methods**:
- `generate`: Generates text for a single prompt.
- `create_completion`: Handles chat-like interactions.

---

#### OpenAPILLM
`OpenAPILLM` interacts with external APIs, such as OpenAI, to generate responses. It uses the OpenAI Python SDK for communication.

**Example Configuration**:
```python
model = OpenAPILLM(
    model_url="https://api.openai.com/v1",
    token="your_api_key",
    default_model="gpt-3.5-turbo"
)
```
