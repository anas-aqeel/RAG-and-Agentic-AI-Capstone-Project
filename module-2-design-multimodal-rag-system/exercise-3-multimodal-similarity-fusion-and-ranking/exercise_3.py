"""
Module 2, Exercise 3: Multimodal Similarity Fusion and Retrieval Ranking

Combines article (text) and image retrieval into a single ranked list via
cross-modal score normalization and weighted late fusion.

No LLM API calls — all models run locally.

Usage:
    1. Run exercise_1.py first to build the vector index
    2. python exercise_3.py

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
# Step 1: Verify vector databases
# =============================================================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run exercise_1.py first to build the index."
    )

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="food_images",          persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images   = image_db._collection.count()

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

# CLIP model (512-d) — used for both image and text→image queries
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


@torch.no_grad()
def embed_query_clip_text(query: str):
    """Encode a text query into CLIP's shared image-text vector space (512-d)."""
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)


print("CLIP embedders ready (ViT-B/32, 512-d)")


# =============================================================================
# Step 3: Utility functions
# =============================================================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def _to_similarity(dists):
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    return 1.0 - np.array(dists, dtype=np.float32)


def _minmax(x):
    """Min-max normalize to [0, 1]; constant arrays → all-ones."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(f"     {snippet}")


# =============================================================================
# Step 4: Retrieval functions
# =============================================================================

def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    """Text → article retrieval using SentenceTransformer (384-d)."""
    q_vec = embed_texts([query])[0]
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):
    """Text → image retrieval using CLIP text encoder (512-d)."""
    q_vec = embed_query_clip_text(query)
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


# =============================================================================
# Step 5: Multimodal fusion
# =============================================================================

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int = 5,
):
    """
    Retrieve from both modalities, normalize scores, apply weighted fusion,
    and return the top-n results sorted by fused score.
    """
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    rows = []
    for j in range(len(t_ids)):
        meta = t_metas[j] if isinstance(t_metas[j], dict) else {}
        rows.append({
            "modality":   "article",
            "id":         meta.get("doc_id", t_ids[j]),
            "cuisine":    meta.get("cuisine", "N/A"),
            "location":   meta.get("location", "N/A"),
            "source":     meta.get("source", "N/A"),
            "text_score": float(t_norm[j]),
            "img_score":  0.0,
            "fused":      float(w_text * t_norm[j]),
            "snippet":    (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        meta = i_metas[j] if isinstance(i_metas[j], dict) else {}
        rows.append({
            "modality":   "image",
            "id":         meta.get("doc_id", i_ids[j]),
            "cuisine":    meta.get("cuisine", "N/A"),
            "location":   meta.get("location", "N/A"),
            "source":     meta.get("source", "N/A"),
            "text_score": 0.0,
            "img_score":  float(i_norm[j]),
            "fused":      float(w_img * i_norm[j]),
            "snippet":    (i_docs[j] or "").replace("\n", " ").strip(),
        })

    rows.sort(key=lambda r: r["fused"], reverse=True)
    if top_n is not None:
        rows = rows[: max(0, int(top_n))]
    return rows


# =============================================================================
# Step 6: Demos
# =============================================================================

print("\n=== Demo 1 — Multimodal fusion (no filters) ===")

q = "cozy noodles with warm atmosphere"
rows = fuse_rank(q, k_text=5, k_img=5, w_text=0.6, w_img=0.4, top_n=5)
print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")
print("Demo 1 complete")


print("\n=== Demo 2 — Multimodal fusion (metadata filters) ===")

q = "handmade pasta and romantic dinner"
rows = fuse_rank(
    q,
    k_text=5, k_img=5,
    w_text=0.6, w_img=0.4,
    where_text={"location": "Pasadena"},
    where_img={"source": "recipe_image"},
    top_n=5,
)
if not rows:
    print("No results with current filters — relaxing to no filter.")
    rows = fuse_rank(q, k_text=5, k_img=5, w_text=0.6, w_img=0.4, top_n=5)
print_fused(rows, title="Demo 2 — Multimodal fusion (metadata filters)")
print("Demo 2 complete")


print("\n=== Demo 3 — Weight tuning ===")

q = "fresh sushi and minimalist presentation"

# 3A — Text-heavy
rows_3a = fuse_rank(q, k_text=5, k_img=5, w_text=0.8, w_img=0.2, top_n=5)
print_fused(rows_3a, title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")

# 3B — Image-heavy
rows_3b = fuse_rank(q, k_text=5, k_img=5, w_text=0.3, w_img=0.7, top_n=5)
print_fused(rows_3b, title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")

print("Demo 3 complete")
print("Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")
