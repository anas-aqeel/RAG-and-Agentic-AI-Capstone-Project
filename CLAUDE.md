# CLAUDE.md — AI Assistant Context for Capstone Project

## Project Identity

This is the **RAG and Agentic AI Capstone Project** from IBM's Professional Certificate on Coursera. The user (Anas Aqeel) is completing all exercises using **Claude via Google Vertex AI** as the LLM backend instead of IBM watsonx.

## LLM Configuration

- **Provider:** Google Vertex AI (Anthropic Claude models)
- **SDK:** `anthropic[vertex]` — use `AnthropicVertex` client
- **Authentication:** `gcloud auth application-default login`
- **Region:** `us-east5`

### Models in Use
| Model | Vertex AI ID | Usage |
|-------|-------------|-------|
| **Sonnet 4.6** | `claude-sonnet-4-6@20260401` | Primary — coding, analysis, most tasks |
| **Haiku 4.5** | `claude-haiku-4-5@20251001` | Simple/fast tasks, iterations |
| **Opus 4.6** | `claude-opus-4-6@20260401` | Complex reasoning only (expensive) |

### Standard Client Setup
```python
import os
from dotenv import load_dotenv
from anthropic import AnthropicVertex

load_dotenv()

client = AnthropicVertex(
    project_id=os.getenv("GCP_PROJECT_ID"),
    region=os.getenv("GCP_REGION", "us-east5")
)

model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6@20260401")
```

## Course Structure — 5 Modules, 12 Exercises + 1 Final Submission

### Module 1: Build a Structured Generative AI Application (4 hrs)
Use LLMs to transform unstructured restaurant data into structured JSON, process multimodal data, build CLI UI.

- **Exercise 1:** Structure Unstructured Restaurant Data with an LLM
  - Input: Raw restaurant descriptions (text)
  - Output: Structured JSON (name, cuisine, location, ratings, etc.)
  - Checklist: Structure Text Data with LLMs

- **Exercise 2:** Process Multimodal Data with LLMs
  - Input: Images + text (customer reviews, food photos)
  - Output: Structured multimodal JSON
  - Checklist: Process Multimodal Customer Data with LLMs

- **Exercise 3:** Build a Command-Line Data Management UI for Restaurant Data
  - Input: Structured JSON from exercises 1-2
  - Output: Interactive CLI for CRUD operations
  - Checklist: Build a Simple Interactive User Interface

### Module 2: Design a Multimodal RAG System (3 hrs)
Build retrieval layer with vector indexes, similarity search, metadata filtering, late-fusion ranking.

- **Exercise 1:** Construct a Multimodal Vector Index
  - Input: Text + image data from Module 1
  - Output: ChromaDB vector index with text/image embeddings
  - Checklist: Multimodal Vector Index Construction

- **Exercise 2:** Similarity Retrieval with Metadata Filtering
  - Input: Queries + vector index
  - Output: Filtered similarity search results
  - Checklist: Similarity Retrieval with Metadata Filtering

- **Exercise 3:** Multimodal Similarity Fusion and Retrieval Ranking
  - Input: Multi-modal query results
  - Output: Late-fusion ranked results
  - Checklist: Multimodal Similarity Fusion and Ranking

### Module 3: Combine Agents into a Multi-Agent System (3 hrs)
Design specialized agents, orchestrate with LangGraph, build Gradio chatbot.

- **Exercise 1:** Design Specialized Agents for a Recommendation System
  - Define agent roles, tools, and prompts
  - Checklist: Define Agents and Their Roles

- **Exercise 2:** Implement and Test a Multi-Agent Recommendation System
  - Orchestrate agents with LangChain/LangGraph
  - Checklist: Integrate Agents into a Multi-Agent System

- **Exercise 3:** Build a Chatbot Interface for the Recommendation System
  - Gradio-based chatbot UI
  - Checklist: Build a Chatbot Interface for the Recommendation System

### Module 4: Integrate Agents, RAG, and Tools with MCP (3 hrs)
Connect everything using Model Context Protocol.

- **Exercise 1:** Build an MCP Server
  - Organize tools, databases, documents in MCP server
  - Checklist: Organize Tools and Data in an MCP Server

- **Exercise 2:** Build an MCP Client
  - Client-server communication layer
  - Checklist: Implement an MCP Client for Server Communication

- **Exercise 3:** Build a Full MCP Application
  - End-to-end LLM-based MCP host
  - Checklist: Design an LLM-based MCP Host

### Module 5: Final Project and Course Wrap-Up (1 hr)
- Submit screenshots and artifacts from all modules
- Graded by AI-based evaluation system

## Workflow — How We Work Together

### What the user provides:
1. The exercise number/name they want to work on
2. The lab instructions or assignment description from Coursera
3. Any specific requirements or constraints

### What Claude generates:
1. Complete, working source code for the exercise
2. Each exercise gets its own folder with its own Python file(s)
3. Code uses Claude via Vertex AI (NOT watsonx, NOT OpenAI)
4. Code follows the lab requirements but adapted for our LLM stack

### File Organization:
```
module-X-.../exercise-Y-.../
├── main.py          # Primary exercise code
├── utils.py         # Helper functions (if needed)
└── README.md        # Exercise-specific notes (only if complex)
```

### Code Standards:
- All LLM calls go through `AnthropicVertex` client
- Use `python-dotenv` for environment variables
- Type hints where practical
- Clear comments explaining non-obvious logic
- Each exercise should be runnable independently
- Exercises within a module can share data (stored in `data/` or passed between exercises)

## Important Notes

- **Credits are limited** — 3 month validity on Vertex AI. Prefer Haiku for testing, Sonnet for production code, Opus only when necessary.
- **Course uses IBM watsonx** in its original labs — we adapt everything to use Claude via Vertex AI instead.
- **Exercises are inter-related** within each module — later exercises build on earlier ones' outputs.
- **Module 5** is just submission — no new code needed, just organize artifacts.

## Tech Stack

| Category | Tool |
|----------|------|
| LLM | Claude (Sonnet 4.6 / Haiku 4.5 / Opus 4.6) via Vertex AI |
| Orchestration | LangChain, LangGraph |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (or Vertex AI embeddings) |
| UI | Gradio (Module 3), CLI (Module 1) |
| Protocol | MCP — Model Context Protocol (Module 4) |
| Data Format | JSON |
| Language | Python 3.10+ |
