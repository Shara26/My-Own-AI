# My Own AI

A Python-based vector database and Retrieval-Augmented Generation (RAG) system built to explore vector search algorithms, similarity metrics, embeddings, and LLM-powered document retrieval.

## Overview

My Own AI combines a custom vector search implementation with a document-based RAG pipeline. It allows documents to be converted into embeddings, stored, and searched using different nearest-neighbor algorithms. The retrieved information can then be passed to an LLM to generate context-aware responses.

The project focuses on understanding the internal concepts behind vector databases and how they are used in modern AI applications.

## Features

* Custom vector database implementation in Python
* Brute-force, KD-Tree, and HNSW search algorithms
* Cosine, Euclidean, and Manhattan distance metrics
* Ollama-based embedding generation
* Document-based Retrieval-Augmented Generation (RAG)
* LLM integration for response generation
* FastAPI REST APIs
* Document management and vector search
* Search benchmarking

## How It Works

The general RAG workflow is:

```text
Documents
   ↓
Embedding Generation
   ↓
Vector Storage
   ↓
User Query
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Documents
   ↓
LLM Context
   ↓
Generated Response
```

The system retrieves relevant information from stored documents and uses it as context for the LLM, helping generate responses grounded in the available data.

## Search Algorithms

The project includes multiple approaches to vector search:

* **Brute Force:** Compares the query vector with stored vectors to identify the nearest results.
* **KD-Tree:** Uses a tree-based structure to organize data for spatial search.
* **HNSW:** Uses a hierarchical graph structure for approximate nearest-neighbor search.

These approaches provide a way to explore the differences between exact and approximate vector search.

## Distance Metrics

The vector database supports:

* **Cosine similarity/distance**
* **Euclidean distance**
* **Manhattan distance**

These metrics determine how the similarity or distance between vectors is calculated during retrieval.

## Tech Stack

* **Python**
* **NumPy**
* **FastAPI**
* **Ollama**
* **HNSW**
* **KD-Tree**
* **RAG**
* **Embeddings**

## Project Structure

```text
My-Own-AI/
│
├── main.py
├── httpLib.py
├── index.html
├── requirements.txt
├── .gitignore
└── README.md
```
