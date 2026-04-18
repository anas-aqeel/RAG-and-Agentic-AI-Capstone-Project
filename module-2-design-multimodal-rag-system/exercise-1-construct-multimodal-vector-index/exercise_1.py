"""
Module 2, Exercise 1: Construct a Multimodal Vector Index

Builds ChromaDB vector indexes for restaurant text data and food images
using SentenceTransformers (text) and CLIP (images).

No LLM API calls — all models run locally.

Usage:
    1. pip install -r requirements.txt
    2. Place structured_restaurant_data.json and augmented_food_recipe.json
       in this directory (outputs from Module 1)
    3. python exercise_1.py

Dependencies from Module 1:
    - structured_restaurant_data.json  (Exercise 1 output)
    - augmented_food_recipe.json       (Exercise 2 output)
"""

import glob
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# =============================================================================
# Step 0: Resolve data files from Module 1
# =============================================================================

M1_EX1_DIR = "../module-1-build-structured-genai-app/exercise-1-structure-text-data-with-llms"
M1_EX2_DIR = "../../module-1-build-structured-genai-app/exercise-2-process-multimodal-customer-data"

RESTAURANT_FILE = "structured_restaurant_data.json"
RECIPE_FILE = "augmented_food_recipe.json"


def resolve_data_file(filename, search_dirs):
    """Find a data file in the current dir or Module 1 directories."""
    if os.path.exists(filename):
        return filename
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            shutil.copy(candidate, filename)
            print(f"  Copied {filename} from {d}")
            return filename
    return None


restaurant_path = resolve_data_file(RESTAURANT_FILE, [M1_EX1_DIR, M1_EX2_DIR])
recipe_path = resolve_data_file(RECIPE_FILE, [M1_EX2_DIR])

if not restaurant_path:
    print(f"ERROR: {RESTAURANT_FILE} not found. Place it in this directory (output from Module 1, Exercise 1).")
    exit(1)
if not recipe_path:
    print(f"ERROR: {RECIPE_FILE} not found. Place it in this directory (output from Module 1, Exercise 2).")
    exit(1)


# =============================================================================
# Step 1: Prepare image dataset
# =============================================================================

ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
ZIP_PATH = "synthetic-recipe-images.zip"
IMG_DIR = "recipe_images"

if not os.path.exists(IMG_DIR):
    if not os.path.exists(ZIP_PATH):
        print("Downloading recipe images...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    print("Extracting images...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(IMG_DIR)

image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
print(f"Images found: {len(image_paths)}")


# =============================================================================
# Step 2: Load structured data
# =============================================================================

print("\n=== Loading Data ===")

with open(RESTAURANT_FILE, "r") as f:
    restaurants = json.load(f)

with open(RECIPE_FILE, "r") as f:
    recipes = json.load(f)

print(f"Loaded restaurants: {len(restaurants)}")
print(f"Loaded recipes:     {len(recipes)}")


# =============================================================================
# Step 3: Initialize embedding models
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
# Step 4: Construct multimodal documents
# =============================================================================

print("\n=== Constructing Documents ===")

# Article documents from restaurant data
article_docs = []
for i, r in enumerate(restaurants):
    name = str(r.get("name", "")).strip()
    if not name:
        continue

    text = (
        f"Restaurant: {name}\n"
        f"Cuisine: {r.get('food_style', '')}\n"
        f"Location: {r.get('location', '')}"
    )

    doc_id = f"rest_{i}"

    article_docs.append(
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": doc_id,
                "cuisine": r.get("food_style"),
                "location": r.get("location"),
                "source": "restaurant",
            },
        )
    )

print(f"Article docs: {len(article_docs)}")

# Image documents from recipe data
image_docs = []
for i, (p, rec) in enumerate(zip(image_paths, recipes)):
    doc_id = f"img_{i}"

    image_docs.append(
        Document(
            page_content=rec.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": doc_id,
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "cuisine": rec.get("cuisine"),
            },
        )
    )

print(f"Image docs: {len(image_docs)}")


# =============================================================================
# Step 5: Construct and persist vector indexes
# =============================================================================

print("\n=== Building Vector Indexes ===")

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if os.path.isdir(DB_DIR):
    shutil.rmtree(DB_DIR)

# Article DB
A = embed_texts([d.page_content for d in article_docs])

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)

print("Article DB ready")

# Image DB
V = embed_images([d.metadata["image_path"] for d in image_docs])

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)

print("Image DB ready")
print("Multimodal Vector Index Construction COMPLETE")

print(f"\nPersisted to: {DB_DIR}")
print(f"  restaurant_articles: {article_db._collection.count()} vectors (384-d)")
print(f"  food_images:         {image_db._collection.count()} vectors (512-d)")
