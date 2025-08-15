# Retrieval Augmented Generation (RAG) with Mini Wikipedia
This project implements a **Retrieval Augmented Generation (RAG)** system designed to answer questions by leveraging external knowledge. 
It demonstrates how to combine efficient information retrieval from a large corpus (mini Wikipedia) with the generative capabilities of a **large language model (LLM)**. 
The aim is to produce more accurate, factual, and contextually relevant answers than an LLM alone might provide, by first retrieving pertinent
information and then using that information to guide the generation process. This system is ideal for tasks
requiring fact-grounded responses from a specific knowledge base.

## Table of Contents

- [Features](#Features)
- [Dataset](#Dataset)
- [Usage](#Usage)

## Features
**Efficient Information Retrieval**: Utilizes `sentence-transformers` to create a vector database from Wikipedia passages, enabling fast and relevant document retrieval.

**Contextualized Language Generation**: Integrates a pre-trained **LLM** `(microsoft/Phi-4-mini-instruct)` to generate answers that are grounded in the retrieved information.

**End-to-End Workflow**: Provides a simple, sequential process for loading data, building the knowledge base, retrieving relevant passages, and generating responses.


## Dataset
**Name**: `rag-datasets/rag-mini-wikipedia`

**Description**:

A curated mini Wikipedia dataset designed for **RAG** experiments.
It contains short Wikipedia passages and is suitable for lightweight training, evaluation, and prototyping of retrieval-augmented models.

**Features**:
- `passages.parquet`: Contains the knowledge base (Wikipedia articles/passages) from which information is retrieved.

- `test.parquet`: Contains a set of questions for evaluating the RAG system.

You can find the dataset at: https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia

## Usage
The core logic is implemented in the `rag_on_mini_wikipedia.ipynb` Jupyter notebook.

### Key Steps in the Notebook
The notebook performs the following operations:

1. **Load Data**: Reads the `passages.parquet` (renamed to `df_infos`) and `test.parquet` (renamed to `df_questions`) file
```python
df_infos = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
df_questions = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
```

2. **Initialize Models**:

  - **Sentence Transformer for Similarity**: Uses `all-mpnet-base-v2` for generating embeddings for retrieval.
