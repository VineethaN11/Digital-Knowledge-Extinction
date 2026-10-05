# Averting Digital Knowledge Extinction: Explainable Agentic Multilingual RAG Framework

A Retrieval-Augmented Generation (RAG) framework for **preserving traditional agricultural knowledge**. It retrieves answers across multiple languages, uses Knowledge Graph reasoning, detects gaps in the knowledge base, and explains how each answer was produced.

## Problem
Traditional agricultural knowledge is often undocumented, spread across languages, and at risk of being lost. This project builds a system that can store this knowledge, retrieve it reliably in different languages, and flag where the knowledge base has gaps.

## Key Features
- **Multilingual semantic retrieval:** queries and documents in different languages are matched by meaning, not keywords
- **Agentic RAG:** multiple components work together to retrieve, reason and generate answers
- **Knowledge Graph reasoning:** relationships between concepts support retrieval and answers
- **Knowledge-gap detection:** identifies queries the knowledge base cannot answer reliably
- **Explainability:** shows the sources and reasoning behind each answer
- **Modular design:** retrieval, reasoning and generation are independent, interoperable modules with their own APIs, including the KGI and TKPS components

## Architecture
```
User Query
    ↓
Multilingual Semantic Retrieval ──→ Vector Database
    ↓
Knowledge Graph Reasoning
    ↓
KGI and TKPS Components
    ↓
Answer Generation + Explanation
    ↓
Knowledge-Gap Detection
```

## Project Structure
```
├── py/                       # Source code (retrieval, reasoning, generation modules)
├── knowledge_corpus/
│   └── Agro dataset/         # Agricultural knowledge documents
├── vector_db/                # Stored embeddings for semantic search
└── evaluation_data/          # Data used to evaluate the system
```

## Tech Stack
- Python
- Embedding-based semantic retrieval with a vector database
- Large Language Models (LLMs)
- Knowledge Graph

## Setup and Run
```bash
# 1. Clone the repo
git clone https://github.com/VineethaN11/Digital-Knowledge-Extinction.git
cd Digital-Knowledge-Extinction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main script from the py/ folder
cd py
```
If the project uses an LLM API, set your API key as an environment variable before running.

## Evaluation
Evaluation data is available in the `evaluation_data/` folder.

## Future Improvements
- Support for more languages and regional dialects
- Larger and more diverse knowledge corpus
- A web interface for farmers and researchers
- Expanded evaluation with more test queries

## Author
**Vineetha N**
[GitHub](https://github.com/VineethaN11) | [LinkedIn](https://www.linkedin.com/in/vineetha-narayan-799111298/)
