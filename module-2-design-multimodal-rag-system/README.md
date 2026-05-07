# Module 2 — Design a Multimodal RAG System

Build the retrieval layer for the recommendation system: vector indexes for text and image data, similarity search with metadata filtering, and late-fusion ranking that combines both modalities.

## Exercises

| # | Folder | What it does |
|---|--------|--------------|
| 1 | [exercise-1-construct-multimodal-vector-index](exercise-1-construct-multimodal-vector-index/) | Builds two ChromaDB collections — text articles via SentenceTransformer (384-d) and food images via CLIP (512-d) |
| 2 | [exercise-2-similarity-retrieval-with-metadata-filtering](exercise-2-similarity-retrieval-with-metadata-filtering/) | Top-k similarity search with optional metadata `where` filters (cuisine, location, etc.) |
| 3 | [exercise-3-multimodal-similarity-fusion-and-ranking](exercise-3-multimodal-similarity-fusion-and-ranking/) | Late-fusion ranking: min-max normalize per-modality scores, then weighted sum |

## Lab source

IBM-provided lab notebooks: [`lab-source/`](lab-source/) (`M2L1_Lab.ipynb`, `M2L2_Lab.ipynb`, `M2L3_Lab.ipynb`).

## Run an exercise

Exercise 1 must run first — it builds the ChromaDB index that Exercises 2 and 3 query. It also auto-pulls upstream data files from Module 1's `data/` folders.

```bash
cd exercise-1-construct-multimodal-vector-index
pip install -r requirements.txt
python exercise_1.py
```

ChromaDB is persisted to `~/chroma_multimodal/` (outside the repo) so multiple exercises can share it.

## Deliverables

- Per-exercise notebooks in `exercise-N-.../submission/exercise_N_submission.ipynb`
- Per-exercise screenshots in `exercise-N-.../submission/screenshots/`
- Final consolidated screenshots in [`/final-submission/answers/`](../final-submission/answers/) (M2L1, M2L2, M2L3)
