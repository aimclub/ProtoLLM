# Protollm-synthetic

This repository contains a set of tools for synthetic observation generation for fine-tuning LLM-based pipelines.

Available pipelines:
- Summarisation
- RAG
- Aspect Summarisation
- Quiz generation
- Free-form generation
- Augmentation of existing dataset

## Installation

```bash
poetry install
```

## Usage

Chains are the main building blocks of the library. They are designed to be used in a pipeline.

```python
from protollm_synthetic.synthetic_pipelines.chains import SummarisationChain
``` 

To run a chain, you need to provide a dataset and a chain.

```python
dataset = Dataset(path="data/sample_summarization_dataset.csv", labels=False)
summarisation_chain = SummarisationChain(llm=llm)
summaries = summarisation_chain.run(dataset, n_examples=100)
``` 