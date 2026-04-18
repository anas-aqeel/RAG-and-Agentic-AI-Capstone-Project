"""
Module 2, Exercise 2: Similarity Retrieval with Metadata Filtering

Performs top-k similarity search over ChromaDB vector indexes built in Exercise 1.
Supports pure similarity search and hybrid similarity + metadata-filter retrieval.

No LLM API calls — all models run locally.

Usage:
    1. Run exercise_1.py first to build the vector index
    2. python exercise_2.py

Dependencies:
    - ~/chroma_multimodal/ vector database (built by exercise_1.py)
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# =============================================================================
# Step 1: Verify vector database
# =============================================================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run exercise_1.py first to build the index."
    )

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

n_articles = article_db._collection.count()
n_images = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun exercise_1.py to rebuild the index."
    )

print(f"Article vectors: {n_articles}")
print(f"Image vectors:   {n_images}")


# =============================================================================
# Step 2: Initialize embedding models
# =============================================================================

print("\n=== Initializing Embedding Models ===")

# Text embedding model (384-d)
text_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


print("Text embedder ready (all-MiniLM-L6-v2, 384-d)")

# Image embedding model (512-d) — CLIP
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()


@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


print("Image embedder ready (CLIP ViT-B/32, 512-d)")


# =============================================================================
# Step 3: Retrieval utilities
# =============================================================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(
            f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} "
            f"| source={source} | distance={dist:.4f}"
        )
        print(f"     {snippet}")


# =============================================================================
# Step 4: Retrieval functions
# =============================================================================

def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    """Similarity retrieval over restaurant articles with optional metadata filtering."""
    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):
    """Similarity retrieval over food images using an image query."""
    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


# =============================================================================
# Step 5: Demos
# =============================================================================

print("\n=== Demo 1 — Article similarity search (no filter) ===")

q = "cozy restaurant with noodles and warm atmosphere"
ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")
print("Demo 1 complete")


print("\n=== Demo 2 — Article similarity search + metadata filter ===")

q = "handmade pasta and romantic dinner"
where_filter = {"location": "Pasadena"}

ids, docs, metas, dists = retrieve_articles(q, k=5, where=where_filter)

if len(ids) == 0:
    print("No results found with current filter — trying without location constraint.")
    ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search (no filter fallback)")
else:
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")

print("Demo 2 complete")


print("\n=== Demo 3 — Image similarity search (image→image) ===")

meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]
QUERY_INDEX = 0  # change to explore different query images

if QUERY_INDEX >= len(meta_all):
    raise ValueError(f"QUERY_INDEX {QUERY_INDEX} out of range (0…{len(meta_all)-1})")

query_img = meta_all[QUERY_INDEX]["image_path"]
print(f"Query image: {query_img}")

# Optional metadata filter — set to None for unfiltered image search
where_img = None

ids, docs, metas, dists = retrieve_images_by_image(query_img, k=5, where=where_img)
print_hits(ids, docs, metas, dists, title="Demo 3 — Image similarity search (image→image)")

print("Demo 3 complete")
print("Similarity Retrieval with Metadata Filtering COMPLETE")
