# Module 1 — Build a Structured Generative AI Application

Transform unstructured restaurant data into structured JSON with an LLM, extend the same pattern to multimodal inputs, and wrap it all in an interactive CLI.

## Exercises

| # | Folder | What it does |
|---|--------|--------------|
| 1 | [exercise-1-structure-text-data-with-llms](exercise-1-structure-text-data-with-llms/) | Reads a paragraph of restaurant prose and emits validated `Restaurant` JSON via Pydantic |
| 2 | [exercise-2-process-multimodal-customer-data](exercise-2-process-multimodal-customer-data/) | Generates Claude vision captions for recipe images and review photos, augmenting both datasets |
| 3 | [exercise-3-build-command-line-data-management-ui](exercise-3-build-command-line-data-management-ui/) | Wraps the Exercise 1 LLM in a CLI that supports browse, view, add, edit, delete, with a backup-on-write safety net |

## Lab source

IBM-provided lab notebooks for this module: [`lab-source/`](lab-source/) (only Lab 2 is provided as a notebook; Labs 1 and 3 are described inline in the Coursera curriculum).

## Run an exercise

```bash
cd exercise-1-structure-text-data-with-llms
pip install -r requirements.txt
python exercise_1.py
```

Outputs land in each exercise's `data/` folder (e.g., `exercise-1-.../data/structured_restaurant_data.json`). Module 2 reads those outputs directly.

## Deliverables

- Per-exercise notebooks in `exercise-N-.../submission/exercise_N_submission.ipynb`
- Per-exercise screenshots in `exercise-N-.../submission/screenshots/`
- Final consolidated screenshots in [`/final-submission/answers/`](../final-submission/answers/) (M1L1, M1L2, M1L3)
