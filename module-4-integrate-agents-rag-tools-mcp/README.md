# Module 4 — Integrate Agents, RAG, and Tools with MCP

Connect the agents, retrieval layer, and tools from Modules 1–3 through the **Model Context Protocol (MCP)** — build a server that exposes resources and tools, a client that consumes them, and an LLM-driven host that orchestrates the whole loop.

> **Note on lab format.** Module 4's labs are delivered as **PDFs** rather than notebooks — see [`lab-source/`](lab-source/). Submission notebooks for these exercises have not been started yet.

## Exercises

| # | Folder | What it does |
|---|--------|--------------|
| 1 | [exercise-1-build-mcp-server](exercise-1-build-mcp-server/) | Builds a FastMCP server that exposes restaurant/recipe data as resources and search/recommend logic as tools |
| 2 | [exercise-2-build-mcp-client](exercise-2-build-mcp-client/) | Implements an MCP client (`ClientSession` over stdio) that connects, lists tools, and invokes them |
| 3 | [exercise-3-build-full-mcp-application](exercise-3-build-full-mcp-application/) | Wires an LLM-driven host that runs a ReAct loop — Reason, Act (call MCP tool), Observe, Repeat |

## Lab source

| File | Lab |
|------|-----|
| `Module-4-Lab-1-Build-MCP-Server.pdf` | Configure tools and data in an MCP server |
| `Module-4-Lab-2-Build-MCP-Client.pdf` | Build and test an MCP client |
| `Module-4-Lab-3-Build-Full-MCP-Application.pdf` | Design an LLM-based MCP host |

> The PDFs were originally delivered as `Module-5-Lab*.pdf` but their content is the Module 4 curriculum (MCP). They were renamed during the project reorganization to align with the curriculum in [`/CLAUDE.md`](../CLAUDE.md) and [`/README.md`](../README.md).

## Prep doc

Long-form study notes for this module live in [`/docs/module-4-preparation.md`](../docs/module-4-preparation.md).

## Deliverables

- Per-exercise submission folders are currently placeholders (`.gitkeep` only)
- Final consolidated screenshots in [`/final-submission/answers/`](../final-submission/answers/) (M4L1, M4L2, M4L3)
