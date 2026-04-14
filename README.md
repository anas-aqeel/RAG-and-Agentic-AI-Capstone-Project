# RAG and Agentic AI Capstone Project

> IBM Professional Certificate — Coursera
> A production-style multimodal RAG system with multi-agent workflows

## Overview

This capstone project demonstrates end-to-end AI system design — from structured data creation to multi-agent deployment. It combines structured data, embeddings, retrieval logic, evaluation strategies, and intelligent workflows into one cohesive solution.

**Key Technologies:** Python, LangChain, LangGraph, ChromaDB, Gradio, MCP (Model Context Protocol)

**LLM Backend:** Claude via Google Vertex AI (Sonnet 4.6, Haiku 4.5, Opus 4.6)

---

## Project Structure

```
Capstone-Project/
├── module-1-build-structured-genai-app/
│   ├── exercise-1-structure-text-data-with-llms/
│   ├── exercise-2-process-multimodal-customer-data/
│   └── exercise-3-build-command-line-data-management-ui/
│
├── module-2-design-multimodal-rag-system/
│   ├── exercise-1-construct-multimodal-vector-index/
│   ├── exercise-2-similarity-retrieval-with-metadata-filtering/
│   └── exercise-3-multimodal-similarity-fusion-and-ranking/
│
├── module-3-combine-agents-multi-agent-system/
│   ├── exercise-1-design-specialized-agents/
│   ├── exercise-2-implement-test-multi-agent-system/
│   └── exercise-3-build-chatbot-interface/
│
├── module-4-integrate-agents-rag-tools-mcp/
│   ├── exercise-1-build-mcp-server/
│   ├── exercise-2-build-mcp-client/
│   └── exercise-3-build-full-mcp-application/
│
├── module-5-final-project/
│   └── final-submission/
│
├── data/                    # Datasets (gitignored, large files)
├── shared/                  # Shared utilities across modules
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── CLAUDE.md                # AI assistant context file
└── README.md                # This file
```

---

## Modules

### Module 1: Build a Structured Generative AI Application
Transform unstructured restaurant descriptions and multimodal data into structured JSON using LLMs, then build a command-line UI.

| Exercise | Lab | Topic |
|----------|-----|-------|
| 1 | Structure Unstructured Restaurant Data with an LLM | Text → structured JSON with LLM |
| 2 | Process Multimodal Data with LLMs | Images + text → structured data |
| 3 | Build a Command-Line Data Management UI | Interactive CLI for restaurant data |

### Module 2: Design a Multimodal RAG System
Build the retrieval layer — vector indexes, similarity search with metadata filtering, and late-fusion ranking.

| Exercise | Lab | Topic |
|----------|-----|-------|
| 1 | Construct a Multimodal Vector Index | Text + image embeddings → vector DB |
| 2 | Similarity Retrieval with Metadata Filtering | Filtered similarity search |
| 3 | Multimodal Similarity Fusion and Retrieval Ranking | Late-fusion ranking across modalities |

### Module 3: Combine Agents into a Multi-Agent System
Design specialized agents, orchestrate them into a recommendation system, and build a Gradio chatbot.

| Exercise | Lab | Topic |
|----------|-----|-------|
| 1 | Design Specialized Agents for a Recommendation System | Agent role definition |
| 2 | Implement and Test a Multi-Agent Recommendation System | Agent orchestration with LangGraph |
| 3 | Build a Chatbot Interface for the Recommendation System | Gradio chatbot UI |

### Module 4: Integrate Agents, RAG, and Tools with MCP
Connect everything using Model Context Protocol — server, client, and full application.

| Exercise | Lab | Topic |
|----------|-----|-------|
| 1 | Build an MCP Server | Organize tools & data in MCP server |
| 2 | Build an MCP Client | Client-server communication |
| 3 | Build a Full MCP Application | End-to-end MCP integration |

### Module 5: Final Project and Course Wrap-Up
Submit the complete capstone — screenshots, artifacts, and documentation of the full pipeline.

---

## Setup

### Prerequisites
- Python 3.10+
- Google Cloud account with Vertex AI enabled
- `gcloud` CLI installed and authenticated

### Installation

```bash
# Clone and enter project
cd "Capstone Project"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Authenticate with Google Cloud
gcloud auth application-default login
```

### Vertex AI Claude Configuration

```python
from anthropic import AnthropicVertex

client = AnthropicVertex(
    project_id="YOUR_GCP_PROJECT_ID",
    region="us-east5"
)
```

---

## LLM Model Selection Guide

| Model | Cost (Input/Output per 1M) | Use When |
|-------|---------------------------|----------|
| Claude Haiku 4.5 | $0.80 / $4.00 | Simple tasks, quick iterations |
| Claude Sonnet 4.6 | $3.00 / $15.00 | Primary model — coding, analysis |
| Claude Opus 4.6 | $15.00 / $75.00 | Complex reasoning only |

---

## Course Info

- **Platform:** [Coursera](https://www.coursera.org/learn/rag-and-agentic-ai-capstone-project)
- **Certificate:** IBM RAG and Agentic AI Professional Certificate
- **Level:** Advanced
- **Duration:** ~1 week at 10 hours/week
- **Assessments:** 16 assignments across all modules
