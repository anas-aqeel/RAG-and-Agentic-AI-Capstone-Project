# Module 3 — Combine Agents into a Multi-Agent System

Design specialized agents (food-style expert, dietary planner, recipe curator, etc.), orchestrate them into a four-phase recommendation workflow with parallel execution, and expose the result through a Gradio chatbot.

## Exercises

| # | Folder | What it does |
|---|--------|--------------|
| 1 | [exercise-1-design-specialized-agents](exercise-1-design-specialized-agents/) | Defines `role`/`goal`/`backstory` configs for each specialist agent (no runnable code — design-only) |
| 2 | [exercise-2-implement-test-multi-agent-system](exercise-2-implement-test-multi-agent-system/) | Implements the four-phase workflow with `ThreadPoolExecutor` for parallel Phase 3 |
| 3 | [exercise-3-build-chatbot-interface](exercise-3-build-chatbot-interface/) | Wraps everything in a Gradio `ChatInterface` with intent classification and preference extraction |

## Lab source

IBM-provided lab notebooks: [`lab-source/`](lab-source/) (`M3L2_Implement_Multi_Agent_Systems.ipynb`, `M3L3_Build_Chatbot_Interface.ipynb`). Exercise 1 is design-only and has no source notebook.

## Run an exercise

Exercise 2 (workflow) and Exercise 3 (chatbot) are independent — neither depends on the other to run.

```bash
cd exercise-3-build-chatbot-interface
pip install -r requirements.txt
python exercise_3.py    # launches Gradio at http://127.0.0.1:7860
```

## Deliverables

- Per-exercise notebooks in `exercise-N-.../submission/exercise_N_submission.ipynb`
- Per-exercise screenshots in `exercise-N-.../submission/screenshots/`
- Final consolidated screenshots in [`/final-submission/answers/`](../final-submission/answers/) (M3L1, M3L2, M3L3)
