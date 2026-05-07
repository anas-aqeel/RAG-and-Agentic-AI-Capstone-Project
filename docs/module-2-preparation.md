# Module 2 — Interview Preparation & Deep Dive

> **What this module covered:** Building a multimodal RAG retrieval layer — embedding text and images into vector spaces, persisting them in ChromaDB, performing similarity search with metadata filtering, and fusing multi-modal results into a single ranked list.

---

## Table of Contents

1. [What is RAG and Why It Exists](#1-what-is-rag-and-why-it-exists)
2. [Vector Embeddings — The Core Idea](#2-vector-embeddings--the-core-idea)
3. [Text Embeddings: SentenceTransformer](#3-text-embeddings-sentencetransformer)
4. [Image Embeddings: CLIP](#4-image-embeddings-clip)
5. [Vector Databases: ChromaDB](#5-vector-databases-chromadb)
6. [Similarity Search & Metadata Filtering](#6-similarity-search--metadata-filtering)
7. [Cross-Modal Retrieval: Text → Images](#7-cross-modal-retrieval-text--images)
8. [Score Normalization](#8-score-normalization)
9. [Late Fusion / Weighted Reranking](#9-late-fusion--weighted-reranking)
10. [Pattern Deep Dives](#10-pattern-deep-dives)
    - [L2 Normalization & Cosine Similarity](#101-l2-normalization--cosine-similarity)
    - [Why Two Separate Collections?](#102-why-two-separate-collections)
    - [Metadata Filtering in Production](#103-metadata-filtering-in-production)
    - [Scaling Vector Search to Millions](#104-scaling-vector-search-to-millions)
    - [Hybrid Search (Vector + BM25)](#105-hybrid-search-vector--bm25)
11. [All Interview Questions & Answers](#11-all-interview-questions--answers)
12. [Concepts You Must Know](#12-concepts-you-must-know)

---

## 1. What is RAG and Why It Exists

### The Problem RAG Solves

LLMs have a knowledge cutoff and can't access private/real-time data. They also hallucinate when asked about things they don't know.

**Analogy:** An LLM without RAG is like a brilliant professor who memorized every textbook published before 2024 but has no internet access. RAG gives them a library card — they can look things up before answering.

### How RAG Works

```
User Query
    ↓
[Embed Query] ──→ vector
    ↓
[Search Vector DB] ──→ top-k relevant documents
    ↓
[Augment Prompt] = "Given these documents: {docs}\n\nAnswer: {query}"
    ↓
[LLM generates answer grounded in retrieved docs]
```

### The Two Phases

| Phase | What happens | When it runs |
|-------|-------------|--------------|
| **Indexing** | Embed all documents → store in vector DB | Once, at build time |
| **Retrieval** | Embed query → search DB → fetch top-k | Every user query |

Module 2 focused entirely on the **indexing** (Ex 1) and **retrieval** (Ex 2-3) phases.

---

## 2. Vector Embeddings — The Core Idea

### What is an Embedding?

An embedding converts something (text, image, audio) into a list of numbers (a vector) such that **similar things end up close together** in vector space.

**Analogy:** Imagine a map of cities. Similar cities (Paris, Lyon) are geographically close. Embeddings create a similar map for meaning — "pizza" and "pasta" are close together, "pizza" and "quantum physics" are far apart.

```
"cozy noodle restaurant" → [0.12, -0.34, 0.87, ..., 0.02]  # 384 numbers
"warm ramen shop"        → [0.11, -0.31, 0.85, ..., 0.04]  # very similar!
"quantum mechanics"      → [-0.78, 0.56, -0.12, ..., 0.91] # very different
```

### Why Numbers?

Computers can't compare "meaning" directly, but they can compute distances between vectors in milliseconds. `np.dot(a, b)` on two 384-d vectors is faster than any text comparison.

### The Geometric Intuition

```
         "Italian food"
              ●
         "pasta"●  ●"pizza"
                  ●"risotto"

                           (far away)
                           ●"JavaScript"
                           ●"React"
```

Documents near the query vector = semantically relevant.

---

## 3. Text Embeddings: SentenceTransformer

### The Model We Used

`all-MiniLM-L6-v2` — a small (22M parameters), fast text embedding model that produces **384-dimensional vectors**.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize → cosine similarity
    ).astype(np.float32)
```

### Key Parameters Explained

| Parameter | What it does | Why we use it |
|-----------|-------------|---------------|
| `batch_size=64` | Process 64 texts at once | Faster than 1-by-1 |
| `normalize_embeddings=True` | Scale each vector to length 1 | Enables cosine similarity with dot product |
| `.astype(np.float32)` | Convert to 32-bit float | Saves memory vs float64, ChromaDB expects it |

### Why MiniLM?

- **Fast:** ~14,000 sentences/second on CPU
- **Small:** 22M params vs BERT-base 110M params
- **Good enough:** Scores ~0.80 on semantic similarity benchmarks
- **Free:** No API calls, runs fully locally

### Interview Question: "Why not use the LLM itself for embeddings?"

Using Claude/GPT for embeddings would cost ~$0.02 per 1M tokens via API vs $0 for a local model. For indexing 100K documents, that's a real cost. Also, embedding models are specifically optimized for similarity (they're trained with contrastive loss), while LLMs are optimized for generation. Specialized tools win in specialized tasks.

---

## 4. Image Embeddings: CLIP

### What is CLIP?

CLIP (Contrastive Language-Image Pre-training, OpenAI 2021) is trained to understand both images and text in a **shared vector space**. The key insight: after training, similar images and their text descriptions land near each other.

```
"a bowl of ramen"  →  [CLIP text encoder]  →  [0.34, -0.21, ..., 0.87]
[photo of ramen]   →  [CLIP image encoder] →  [0.33, -0.22, ..., 0.86]
                                               ↑ nearly identical vectors!
```

### How We Used CLIP

```python
from transformers import CLIPModel, CLIPProcessor
import torch

device = "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_model.eval()

# Embed images
@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)   # (B, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

# Embed text query for image search (text → image space)
@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)        # (1, 512)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)
```

### Why `@torch.no_grad()`?

During inference, we don't need gradients (those are for training). Disabling them:
- Saves ~50% memory
- Runs ~20% faster

**Analogy:** You wouldn't record a practice run with professional cameras just to check if your shoes fit.

### CLIP Architecture

```
Image ──→ [ViT-B/32 image encoder] ──→ 512-d vector
                                              ↕ cosine similarity
Text  ──→ [Transformer text encoder] ──→ 512-d vector
```

ViT-B/32 means: Vision Transformer, Base size, patch size 32×32 pixels.

### Why batch_size=16 for images vs 64 for text?

Images are much larger tensors. A single 224×224 RGB image = 150,528 numbers. A sentence = ~20 tokens. Processing 16 images at once uses roughly the same memory as 64 short sentences.

---

## 5. Vector Databases: ChromaDB

### What is ChromaDB?

A lightweight, open-source vector database. Think of it as SQLite but for vectors — stores embeddings + metadata, supports fast similarity search.

```python
from langchain_chroma import Chroma
from pathlib import Path

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

# Create/open a collection
db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,  # saves to disk
)

# Insert vectors
db._collection.upsert(
    ids=["rest_0", "rest_1", "rest_2"],        # unique IDs
    embeddings=[[0.1, 0.2, ...], [...]],       # the actual vectors
    documents=["Restaurant: Sakura...", ...],  # text content
    metadatas=[{"cuisine": "Japanese", "location": "Pasadena"}, ...],
)
```

### Key Operations

| Operation | Method | Purpose |
|-----------|--------|---------|
| Insert/update | `upsert()` | Add docs (idempotent — safe to re-run) |
| Count | `_collection.count()` | Verify data was written |
| Search | `_collection.query()` | Find similar vectors |
| Get all metadata | `_collection.get()` | Inspect stored records |

### Why Two Separate Collections?

We built `restaurant_articles` (384-d) and `food_images` (512-d) as separate collections because:

1. **Different dimensions** — ChromaDB collections enforce a fixed vector size. You can't mix 384-d and 512-d vectors in one collection.
2. **Different semantics** — Text vectors and image vectors exist in different learned spaces. Cross-collection comparison requires normalization (Exercise 3).
3. **Independent retrieval** — You can query just articles OR just images, then fuse.

### Why `upsert` instead of `add`?

`add()` throws an error if an ID already exists. `upsert()` inserts or updates — safe to run multiple times without duplicates. Critical for re-runs and pipeline restarts.

---

## 6. Similarity Search & Metadata Filtering

### The Query Flow

```python
def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    # 1. Embed the query into the same vector space as the index
    q_vec = embed_texts([query])[0]  # 384-d

    # 2. Search ChromaDB
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,            # optional metadata filter
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)
```

### What is `distance` in ChromaDB?

ChromaDB returns **L2 distance** by default (but we used cosine-normalized vectors, so L2 distance ≈ 2 × (1 - cosine_similarity)).

```
distance = 0.0  →  identical vectors (cosine similarity = 1.0)
distance = 1.0  →  orthogonal (cosine similarity = 0.5)
distance = 2.0  →  opposite (cosine similarity = 0.0)
```

We convert to similarity: `similarity = 1 - distance`

### Metadata Filtering

Chroma's `where` parameter uses MongoDB-style filter syntax:

```python
# Exact match
where = {"location": "Pasadena"}

# Multiple conditions (AND)
where = {"$and": [{"location": "Pasadena"}, {"cuisine": "Italian"}]}

# Multiple values (OR / IN)
where = {"location": {"$in": ["Pasadena", "Burbank", "Glendale"]}}

# Not equal
where = {"source": {"$ne": "restaurant"}}
```

### Filter-First vs Post-Filter

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Pre-filter** (what Chroma does) | Filter candidates first, then rank by similarity | Fast — searches smaller space | May miss results if filter is too strict |
| **Post-filter** | Retrieve top-1000 by similarity, then filter | More recall | Slower, must retrieve more than you need |

**Analogy:** Pre-filter = ask the librarian to only bring books from the Science section, then pick the most relevant. Post-filter = ask for the 1000 most relevant books from the whole library, then keep only the Science ones.

---

## 7. Cross-Modal Retrieval: Text → Images

### The CLIP Magic

Because CLIP was trained to align text and image representations, a text query can search the image database:

```python
# User types: "fresh sushi and minimalist presentation"
# → CLIP text encoder → 512-d vector
# → Search food_images collection (which contains CLIP image vectors)
# → Returns visually similar images!

q_vec = embed_query_clip_text("fresh sushi and minimalist presentation")
res = image_db._collection.query(query_embeddings=[q_vec.tolist()], n_results=5)
```

This is called **zero-shot cross-modal retrieval** — no fine-tuning needed, works out-of-the-box because CLIP's training objective was exactly this alignment.

### Text-only model can't do this

SentenceTransformer only produces text embeddings. If you tried to query the image database using a SentenceTransformer vector (384-d vs 512-d), ChromaDB would error: dimension mismatch.

```
CLIP text query (512-d)  ──→  image DB (512-d)  ✓  works!
MiniLM text query (384-d) ──→ image DB (512-d)  ✗  dimension mismatch
MiniLM text query (384-d) ──→ article DB (384-d) ✓  works!
```

---

## 8. Score Normalization

### The Problem

Two retrieval pipelines return scores on different scales:

```
Article DB (MiniLM): distances ≈ [0.05, 0.12, 0.31, 0.45, 0.60]
Image DB (CLIP):     distances ≈ [0.18, 0.24, 0.35, 0.41, 0.55]
```

MiniLM scores cluster differently than CLIP scores. You can't directly compare `0.12` from text with `0.12` from images — they were produced by different models with different distributions.

### The Solution: Min-Max Normalization

```python
def _minmax(x):
    """Normalize array to [0, 1] range."""
    x = np.array(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal → full confidence
    return (x - lo) / (hi - lo)

# Before normalization:
# text_sims  = [0.95, 0.88, 0.69, 0.55, 0.40]
# image_sims = [0.82, 0.76, 0.65, 0.59, 0.45]

# After normalization:
# text_norm  = [1.00, 0.87, 0.52, 0.27, 0.00]
# image_norm = [1.00, 0.84, 0.54, 0.38, 0.00]
# Both now on [0, 1] — comparable!
```

**Analogy:** Grading students on different scales. One teacher uses 0-100, another uses 0-10. Before ranking students across both classes, normalize both to 0-1.

### Why Not Z-Score Normalization?

Z-score (`(x - mean) / std`) can produce negative values, which breaks intuitive weighting. Min-max keeps scores in [0, 1], making fusion weights (`w_text=0.6, w_img=0.4`) directly interpretable as percentages.

---

## 9. Late Fusion / Weighted Reranking

### What is Late Fusion?

Each modality retrieves independently, then scores are combined into one ranking. Called "late" because fusion happens after retrieval (as opposed to early fusion where you embed different modalities together).

```
Query: "cozy noodles warm atmosphere"
        ↓                    ↓
  [Text retrieval]    [Image retrieval]
  5 article results   5 image results
        ↓                    ↓
  normalize scores    normalize scores
        ↓                    ↓
         [Weighted fusion]
         fused = 0.6 × text_norm + 0.4 × img_norm
                 ↓
        [Sort by fused score]
                 ↓
        Top-5 unified ranking
```

### Our Implementation

```python
def fuse_rank(query, k_text=5, k_img=5, w_text=0.6, w_img=0.4, top_n=5):
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img)

    # Normalize within each modality independently
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build one pool with fused scores
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality": "article",
            "fused": w_text * t_norm[j],   # text contributes 60%
            ...
        })
    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "fused": w_img * i_norm[j],    # image contributes 40%
            ...
        })

    rows.sort(key=lambda r: r["fused"], reverse=True)
    return rows[:top_n]
```

### Effect of Weight Tuning

| Weights | Top results | Best when |
|---------|-------------|-----------|
| `w_text=0.8, w_img=0.2` | Mostly articles | User wants venue info, business details |
| `w_text=0.5, w_img=0.5` | Balanced mix | General discovery |
| `w_text=0.3, w_img=0.7` | Mostly images | User wants visual inspiration, food photos |

**Real-world example:** A restaurant app might boost `w_text` when the user is asking "where can I find..." and boost `w_img` when the user is asking "show me dishes like this...".

### Why Not Just Merge and Re-embed?

Early fusion (combining text+image into one embedding) requires a cross-modal model trained on paired data. Our approach (late fusion) works with any two embedding models regardless of dimension and doesn't require retraining.

---

## 10. Pattern Deep Dives

### 10.1 L2 Normalization & Cosine Similarity

**The math:**

Cosine similarity = `dot(a, b) / (||a|| × ||b||)`

If both vectors are already L2-normalized (length = 1):
`||a|| = ||b|| = 1` → cosine similarity = `dot(a, b)`

So by normalizing at index time AND at query time, we get cosine similarity for free with just a dot product.

```python
# At indexing time:
feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize each vector to length 1

# At query time:
q_vec = embed_texts([query])[0]  # also normalized (normalize_embeddings=True)

# Similarity = just a dot product
similarity = np.dot(q_vec, stored_vec)  # ranges from -1 to 1
```

**ChromaDB stores L2 distance**, not similarity. But since both vectors are normalized:
```
L2_dist(a, b)² = ||a - b||² = ||a||² + ||b||² - 2·dot(a,b)
               = 1 + 1 - 2·cosine_sim
               = 2·(1 - cosine_sim)

∴ cosine_sim = 1 - L2_dist²/2  ≈  1 - L2_dist  (for small distances)
```

We use the approximation: `similarity = 1 - distance`

### 10.2 Why Two Separate Collections?

```
restaurant_articles  →  384-d (MiniLM space, text semantics)
food_images          →  512-d (CLIP space, visual+text alignment)
```

**You cannot mix dimensions in one collection.** ChromaDB enforces that all vectors in a collection have the same dimensionality (it creates an ANN index at fixed dimension).

**Design insight:** Separate collections also enable independent scaling — if you add 10M images, the article index is unaffected. They can be on different machines in a distributed system.

### 10.3 Metadata Filtering in Production

Production RAG systems typically need both semantic relevance AND business constraints:

```python
# E-commerce: only show in-stock items
where = {"$and": [{"in_stock": True}, {"category": "electronics"}]}

# News app: only recent articles
where = {"published_date": {"$gte": "2025-01-01"}}

# Our restaurant app: location + cuisine
where = {"$and": [{"location": "Pasadena"}, {"cuisine": "Italian"}]}
```

**Pitfall:** Overly strict filters return zero results. Production systems handle this with fallback:

```python
results = retrieve(query, where=strict_filter)
if not results:
    results = retrieve(query, where=relaxed_filter)  # try looser constraint
if not results:
    results = retrieve(query, where=None)  # no filter, pure similarity
```

### 10.4 Scaling Vector Search to Millions

ChromaDB uses **HNSW** (Hierarchical Navigable Small World) index for approximate nearest neighbor search.

**Analogy:** Finding a friend in a city. Brute force = knock on every door. HNSW = ask your closest friend, who asks their closest friend, navigating the social graph until you find the target.

| Scale | Tool | Strategy |
|-------|------|----------|
| < 100K vectors | ChromaDB (local) | Exact or HNSW |
| 100K – 10M | Pinecone, Weaviate, Qdrant | Managed ANN index |
| 10M+ | Faiss + custom infra | GPU-accelerated HNSW/IVF |

```python
# ChromaDB HNSW settings (tunable)
db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
    collection_metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # higher = better recall, slower build
        "hnsw:search_ef": 100,        # higher = better recall, slower query
    }
)
```

### 10.5 Hybrid Search (Vector + BM25)

Pure vector search misses **exact keyword matches**. Hybrid search combines:

- **Dense retrieval** (vector similarity) — semantic understanding
- **Sparse retrieval** (BM25/TF-IDF) — keyword matching

```python
# Conceptual hybrid search
dense_results  = vector_db.query(q_vec, k=50)
sparse_results = bm25_index.search(query_text, k=50)

# Reciprocal Rank Fusion (RRF) — a common fusion method
def rrf_score(rank, k=60):
    return 1 / (k + rank)

fused = {}
for rank, doc_id in enumerate(dense_results):
    fused[doc_id] = fused.get(doc_id, 0) + rrf_score(rank)
for rank, doc_id in enumerate(sparse_results):
    fused[doc_id] = fused.get(doc_id, 0) + rrf_score(rank)

final = sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

**When to use:** When users include specific names, codes, or rare keywords that embeddings might not capture well.

---

## 11. All Interview Questions & Answers

### Q1: "What is a vector embedding and why do we use it?"

**Answer:**
A vector embedding is a dense numerical representation of data (text, image, audio) where semantically similar items have numerically similar vectors (high cosine similarity). We use them because:
1. Computers can compute vector distance in microseconds
2. They capture semantic meaning, not just keyword overlap
3. They enable cross-modal comparison (text query → image results)

**Code to mention:**
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(["cozy noodle restaurant"], normalize_embeddings=True)
# vecs[0] is a 384-d float32 array
```

---

### Q2: "Why normalize embeddings to L2 length 1?"

**Answer:**
Normalizing converts any distance metric to cosine similarity via a simple dot product. Without normalization, longer documents produce larger raw vector values that dominate similarity scores regardless of meaning. After L2 normalization:
- `dot(a, b)` = cosine similarity directly
- All vectors lie on the unit hypersphere — distance is purely about angle (direction/meaning), not magnitude (length/word count)

---

### Q3: "Why did you use separate embedding models for text and images?"

**Answer:**
Text and images are different modalities requiring specialized encoders:
- **SentenceTransformer** is trained specifically for semantic text similarity via contrastive sentence pairs
- **CLIP** is trained on 400M image-text pairs to align visual and linguistic representations in a shared space

Using CLIP for text-only retrieval would waste its alignment capability. Using SentenceTransformer for image retrieval is impossible — it can't process pixel data.

---

### Q4: "How does ChromaDB persist data? What happens if the process crashes?"

**Answer:**
ChromaDB with `persist_directory` writes to disk using SQLite + parquet files. The data survives process restart. On crash, the database is recoverable — SQLite provides ACID transactions so partial writes don't corrupt existing data.

In our pipeline, we call `upsert()` which is idempotent — re-running after a crash inserts new records without duplicating existing ones.

---

### Q5: "How does metadata filtering interact with vector search in ChromaDB?"

**Answer:**
ChromaDB applies metadata filters **before** similarity search (pre-filtering). It first narrows the candidate pool using the `where` clause, then runs ANN search only on the filtered subset.

This is efficient but can return zero results if the filter is too strict. In production, implement a fallback:
```python
results = query(where=strict)
if not results:
    results = query(where=None)
```

---

### Q6: "Why can't you directly compare similarity scores from the text and image collections?"

**Answer:**
Different embedding models produce scores on different scales with different distributions:
- MiniLM might produce distances clustered between 0.1-0.5
- CLIP might produce distances clustered between 0.2-0.8

A score of 0.3 from MiniLM and 0.3 from CLIP don't mean the same level of relevance. Before fusion, we min-max normalize each modality's scores to [0,1] independently so they become comparable.

---

### Q7: "What is late fusion and why use it instead of early fusion?"

**Answer:**
- **Early fusion:** Combine multiple modalities before encoding (e.g., concatenate image + text into one encoder)
- **Late fusion:** Encode each modality independently, retrieve separately, combine scores afterward

We use late fusion because:
1. No cross-modal model needed — any two embedding models work
2. Each modality's index is independently updatable
3. Weights are transparent and tunable at runtime
4. Different modalities can be queried selectively

---

### Q8: "How would you scale this system to 10 million documents?"

**Answer:**
1. **Replace ChromaDB** with Pinecone/Qdrant (distributed ANN index)
2. **Batch embedding:** Process in chunks of 1,000, use GPU for CLIP
3. **Async indexing pipeline:** Use Celery/Redis queue for background embedding
4. **Tiered storage:** Recent/popular documents in fast HNSW index, older in slower flat index
5. **Caching:** Cache embeddings for repeated content (hash-based)
6. **Sharding:** Split collections by date, geography, or category

```
10M docs × 512-d float32 = ~20GB RAM for full in-memory index
Pinecone handles this with pod-based scaling
```

---

### Q9: "What's the difference between image→image and text→image retrieval?"

**Answer:**
Both use CLIP's image encoder (512-d), but they differ in how the query is encoded:

| Type | Query encoder | Use case |
|------|--------------|---------|
| image→image | CLIP image encoder | "Find dishes visually similar to this photo" |
| text→image | CLIP text encoder | "Find dishes that look like fresh sushi" |

Both produce 512-d vectors in CLIP's shared space — the same image collection handles both. CLIP was explicitly trained for text-image alignment, so text→image works zero-shot without fine-tuning.

---

### Q10: "If a metadata filter returns no results, how should the system respond?"

**Answer:**
Implement a graceful degradation strategy:
```python
# Try progressively relaxed constraints
for where in [strict_filter, partial_filter, None]:
    results = retrieve(query, where=where)
    if results:
        break

# Optionally tell the user what constraint was relaxed
if where != strict_filter:
    print(f"Note: showing results without location constraint")
```

In production, also log these fallback events — frequent fallbacks indicate your data doesn't match user expectations (e.g., the filter value "Pasadena" exists in user queries but not in the data).

---

## 12. Concepts You Must Know

### Quick Reference Table

| Concept | One-line explanation |
|---------|---------------------|
| **Embedding** | Data → dense vector; similar items → nearby vectors |
| **SentenceTransformer** | Local text embedding model, 384-d, no API needed |
| **CLIP** | Image+text model trained on alignment; enables text→image search |
| **ChromaDB** | Lightweight vector DB; stores vectors + metadata; persists to disk |
| **L2 normalization** | Scale vector to length 1 so dot product = cosine similarity |
| **Cosine similarity** | Angle between vectors; 1=identical, 0=orthogonal, -1=opposite |
| **L2 distance** | Geometric distance; ≈ 2×(1 − cosine_sim) for normalized vectors |
| **upsert** | Insert-or-update; idempotent; safe to re-run |
| **Metadata filtering** | Constrain search by structured fields (location, cuisine, etc.) |
| **Pre-filtering** | Filter candidates before ANN search (Chroma's default) |
| **Top-k retrieval** | Return k most similar documents |
| **Cross-modal retrieval** | Query text → search image index using CLIP's shared space |
| **Min-max normalization** | Scale values to [0,1] so different models are comparable |
| **Late fusion** | Retrieve per modality → normalize → weighted combine → rerank |
| **Fusion weight** | `w_text=0.6, w_img=0.4` → text contributes 60% of fused score |
| **HNSW** | Graph-based ANN index used by ChromaDB; fast approximate search |
| **ANN** | Approximate Nearest Neighbor — fast but may miss exact top-k |

### The Pipeline in One Diagram

```
                     INDEXING (once)
                     ──────────────
Restaurant JSON ─→ embed_texts() ─→ 384-d vectors ─→ restaurant_articles DB
Recipe Images   ─→ embed_images() ─→ 512-d vectors ─→ food_images DB


                     RETRIEVAL (every query)
                     ──────────────────────
User Query "cozy noodles"
      │
      ├──→ embed_texts(query) ──→ search restaurant_articles ──→ top-5 articles
      │                                                                │
      └──→ embed_query_clip_text(query) ──→ search food_images ──→ top-5 images
                                                                       │
                                         both streams ─→ normalize ─→ fuse ─→ top-5 unified
```

### The 3 Key Distance→Similarity Conversions

```python
# Step 1: ChromaDB returns distance (smaller = more similar)
distances = [0.05, 0.18, 0.35, 0.44, 0.62]

# Step 2: Convert to similarity (larger = more similar)
similarities = [1 - d for d in distances]  # [0.95, 0.82, 0.65, 0.56, 0.38]

# Step 3: Normalize to [0, 1] for cross-modal fusion
lo, hi = min(similarities), max(similarities)
normalized = [(s - lo) / (hi - lo) for s in similarities]  # [1.0, 0.77, 0.47, 0.32, 0.0]
```
