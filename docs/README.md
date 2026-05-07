# Preparation Docs

Long-form study notes covering each module — concepts, code patterns, interview questions, real-world scenarios, and production concerns. Each module's prep doc is provided in both Markdown (source) and HTML (rendered for reading).

| File | Module | Topics covered |
|------|--------|----------------|
| [module-1-preparation.md](module-1-preparation.md) / [.html](module-1-preparation.html) | Module 1 — Build a Structured GenAI App | Structured-output prompting, JSON validation, multimodal LLMs, CLI design |
| [module-2-preparation.md](module-2-preparation.md) / [.html](module-2-preparation.html) | Module 2 — Design a Multimodal RAG System | Embeddings, ChromaDB, cosine similarity, metadata filtering, late-fusion ranking |
| [module-3-preparation.md](module-3-preparation.md) / [.html](module-3-preparation.html) | Module 3 — Combine Agents into a Multi-Agent System | Agent design, LangGraph, parallel execution, intent classification, Gradio |
| [module-4-preparation.md](module-4-preparation.md) / [.html](module-4-preparation.html) | Module 4 — Integrate with MCP | MCP host/client/server, FastMCP, ReAct loop, sampling, roots, transports |

> Module 5 (Final Project) has no separate prep doc — submission instructions live in [`module-5-final-project/README.md`](../module-5-final-project/README.md) and the consolidated artifacts live in [`final-submission/`](../final-submission/).

## Regenerating the HTML

The HTML files are generated from the MD source using [`.scripts/md_to_docs_html.py`](../.scripts/md_to_docs_html.py). After editing any prep doc:

```bash
python .scripts/md_to_docs_html.py
```
