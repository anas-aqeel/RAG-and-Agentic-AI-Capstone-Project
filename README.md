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
├── README.md                          # This file
├── CLAUDE.md                          # AI assistant context
├── requirements.txt                   # Top-level Python dependencies
├── .env.example                       # Environment variable template
│
├── docs/                              # All preparation / study notes
│   ├── module-1-preparation.{md,html}
│   ├── module-2-preparation.{md,html}
│   ├── module-3-preparation.{md,html}
│   └── module-4-preparation.{md,html}
│
├── final-submission/                  # Capstone deliverable gallery
│   ├── questions/                     # 12 IBM-provided question screenshots
│   └── answers/                       # 12 user-generated answer screenshots
│
├── shared/                            # Shared utilities across exercises
│
├── module-1-build-structured-genai-app/
│   ├── README.md                      # Module overview + exercise table
│   ├── lab-source/                    # IBM-provided lab notebooks
│   ├── exercise-1-structure-text-data-with-llms/
│   ├── exercise-2-process-multimodal-customer-data/
│   └── exercise-3-build-command-line-data-management-ui/
│
├── module-2-design-multimodal-rag-system/         # (same shape as Module 1)
├── module-3-combine-agents-multi-agent-system/    # (same shape)
├── module-4-integrate-agents-rag-tools-mcp/       # lab-source contains PDFs (not notebooks)
└── module-5-final-project/                        # Submission checklist only
```

**Every exercise folder follows the same template:**

```
exercise-N-name/
├── exercise_N.py                      # Local Vertex AI implementation
├── requirements.txt                   # Pinned dependencies
├── data/                              # Input/output files (omitted if none)
└── submission/
    ├── exercise_N_submission.ipynb    # IBM-format submission notebook
    └── screenshots/                   # Answer screenshots for grading
```

## Where things live

- **Course material that came from IBM** lives in `module-N-.../lab-source/` (notebooks for Modules 1–3, PDFs for Module 4).
- **The user's local Claude/Vertex implementations** live at the exercise root (`exercise_N.py`) and read/write `./data/`.
- **The user's IBM-format submissions** live in `module-N-.../exercise-N-.../submission/` and use the lab's default in-cwd paths (so they run unchanged in IBM's grader).
- **Preparation/study docs** for every module live in [`docs/`](docs/) — both Markdown source and rendered HTML.
- **The final capstone deliverable** (12 question + 12 answer screenshots) is consolidated in [`final-submission/`](final-submission/).

## Reuse this template

To fork this layout for a different course:

1. Replace the `module-N-...` folder names with your own course's modules.
2. Replace each exercise's `exercise_N.py` with your implementation; keep the `data/` + `submission/` shape.
3. Drop your lab notebooks/PDFs into `module-N-.../lab-source/`.
4. Write your prep docs as Markdown files in `docs/` and run [`.scripts/md_to_docs_html.py`](.scripts/md_to_docs_html.py) to render them to styled HTML.
5. Update this README and each module's README with your topics.

The structure intentionally separates **provided material** (lab-source/) from **your work** (exercise root + submission/) so that a fork can be cleaned and reused without confusion about who wrote what.

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
