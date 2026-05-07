"""
Module 1, Exercise 2: Process Multimodal Data with LLMs
Standalone script using Claude via Google Vertex AI

This exercise:
  - Loads food recipe data + images, generates captions with Claude's vision
  - Loads user review data + image URLs, generates review-context captions
  - Saves augmented JSON files for downstream use

Usage:
    1. pip install -r requirements.txt
    2. gcloud auth application-default login
    3. Ensure .env exists at project root with GCP_PROJECT_ID
    4. python exercise_2.py
"""

import os
import json
import ast
import base64
import zipfile
import urllib.request
from io import BytesIO

import requests
from PIL import Image
from dotenv import load_dotenv
from anthropic import AnthropicVertex
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(dotenv_path="../../.env")


# =============================================================================
# Data URLs
# =============================================================================

RECIPES_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json"
REVIEWS_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json"
IMAGES_ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"

DATA_DIR = "data"
RECIPES_FILE = os.path.join(DATA_DIR, "Recipes.json")
REVIEWS_FILE = os.path.join(DATA_DIR, "Synthetic-User-Reviews.json")
IMAGES_ZIP = os.path.join(DATA_DIR, "synthetic-recipe-images.zip")
IMAGES_DIR = os.path.join(DATA_DIR, "synthetic_recipe_images")
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# Step 0: Download data if not present
# =============================================================================

def download_if_missing(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  -> Saved {filepath}")

download_if_missing(RECIPES_URL, RECIPES_FILE)
download_if_missing(REVIEWS_URL, REVIEWS_FILE)
download_if_missing(IMAGES_ZIP_URL, IMAGES_ZIP)

if not os.path.exists(IMAGES_DIR):
    print("Extracting recipe images...")
    with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"  -> Extracted to {IMAGES_DIR}/")


# =============================================================================
# Step 1: Load and explore the data
# =============================================================================

print("\n=== Loading Data ===")

with open(RECIPES_FILE, "r") as f:
    recipe_data = json.load(f)

with open(REVIEWS_FILE, "r") as f:
    user_review_data = json.load(f)

print(f"Recipes: {len(recipe_data)}")
print(f"User reviews: {len(user_review_data)}")

# Show first recipe structure
print("\n--- First Recipe ---")
for key, value in recipe_data[0].items():
    print(f"  {key} ({type(value).__name__}): {value}")

# Show first review structure
print("\n--- First Review ---")
for key, value in user_review_data[0].items():
    print(f"  {key} ({type(value).__name__}): {value}")


# =============================================================================
# Step 2: Define the Vision LLM (Claude via Vertex AI)
# =============================================================================

vertex_client = AnthropicVertex(
    project_id=os.getenv("GCP_PROJECT_ID"),
    region=os.getenv("GCP_REGION", "us-east5"),
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5@20251001")


def vision_llm(system_msg, prompt_txt, image_source):
    """
    Call Claude with vision capabilities.

    image_source: either a local file path (str) or raw bytes.
    """
    if isinstance(image_source, str):
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    else:
        image_bytes = image_source

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Detect media type from first bytes
    if image_bytes[:8].startswith(b"\x89PNG"):
        media_type = "image/png"
    elif image_bytes[:2] == b"\xff\xd8":
        media_type = "image/jpeg"
    else:
        media_type = "image/png"  # default for this dataset

    message = vertex_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=system_msg,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_txt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )
    return message.content[0].text


# Test vision LLM on first recipe image
print("\n=== Testing Vision LLM ===")
test_image_path = os.path.join(IMAGES_DIR, f"recipe{recipe_data[0]['id']}.png")
test_response = vision_llm(
    "You are a helpful assistant.",
    "What food is shown in this image? Answer in one sentence.",
    test_image_path,
)
print(f"Test caption: {test_response}")


# =============================================================================
# Step 3: Prompt templates
# =============================================================================

def image_caption_prompt_template(food_name):
    """Prompt for captioning recipe food images."""
    system_msg = (
        "You are a professional food photographer and culinary expert. "
        "Your task is to generate concise, accurate, and descriptive captions "
        "for food images. Focus on visible ingredients, cooking style, "
        "presentation, colors, and textures."
    )
    prompt_txt = (
        f"Please describe this image of {food_name}. "
        "Include details about the visible ingredients, cooking method, "
        "presentation style, colors, textures, and any garnishes. "
        "Keep the description concise and informative in 2-3 sentences."
    )
    return system_msg, prompt_txt


def review_context_image_caption_prompt_template(reviews):
    """Prompt for captioning user review images with review context."""
    system_msg = (
        "You are a culinary expert and food critic. "
        "Your task is to generate concise image descriptions for food images "
        "that align with the context and sentiment expressed in customer reviews. "
        "Focus on visual details such as food presentation, ingredients, colors, "
        "and textures that match the experience described in the reviews."
    )
    prompt_txt = (
        f'Based on the following customer review(s): "{reviews}", '
        "please describe this food image in 2-3 sentences. "
        "Highlight visual details that are consistent with the reviewer's "
        "experience and sentiment."
    )
    return system_msg, prompt_txt


# =============================================================================
# Exercise 1: Caption all recipe images and augment recipe data
# =============================================================================

print("\n=== Exercise 1: Captioning Recipe Images ===")

for i, recipe in enumerate(recipe_data):
    food_name = recipe["name"]
    image_path = os.path.join(IMAGES_DIR, f"recipe{recipe['id']}.png")

    if not os.path.exists(image_path):
        print(f"  [{i}] Image missing: {image_path}, skipping")
        recipe["image_description"] = ""
        continue

    sys_msg, prompt_txt = image_caption_prompt_template(food_name)
    caption = vision_llm(sys_msg, prompt_txt, image_path)
    recipe["image_description"] = caption

    if (i + 1) % 20 == 0:
        print(f"  {i + 1} out of {len(recipe_data)} done")

print(f"ALL DONE! Captioned {len(recipe_data)} recipes.")

# Save augmented recipe data
with open(os.path.join(DATA_DIR, "augmented_food_recipe.json"), "w", encoding="utf-8") as f:
    json.dump(recipe_data, f, indent=4)
print("Saved data/augmented_food_recipe.json")


# =============================================================================
# Exercise 2: Caption user review images and augment review data
# =============================================================================

print("\n=== Exercise 2: Captioning User Review Images ===")


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url):
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response


for i, review in enumerate(user_review_data):
    review_images = ast.literal_eval(review["images"])
    review_image_captions = []

    if review_images:
        review_text = review.get("text", "")
        sys_msg, prompt_txt = review_context_image_caption_prompt_template(review_text)

        for img_url in review_images:
            try:
                img_response = get_data_with_retry(img_url)
                caption = vision_llm(sys_msg, prompt_txt, img_response.content)
                review_image_captions.append(caption)
                print(f"  [{i}] Captioned image from review {review['reviewId']}")
            except Exception as e:
                print(f"  [{i}] FAILED for {img_url}: {e}")
                review_image_captions.append("")

    review["image_captions"] = review_image_captions

print(f"ALL DONE! Processed {len(user_review_data)} reviews.")

# Save augmented review data
with open(os.path.join(DATA_DIR, "augmented_user_review.json"), "w", encoding="utf-8") as f:
    json.dump(user_review_data, f, indent=4)
print("Saved data/augmented_user_review.json")


# =============================================================================
# Summary
# =============================================================================

print("\n=== Summary ===")
print(f"Recipes with captions: {sum(1 for r in recipe_data if r.get('image_description'))}")
print(f"Reviews with image captions: {sum(1 for r in user_review_data if r.get('image_captions'))}")
print("\nOutput files:")
print("  - data/augmented_food_recipe.json")
print("  - data/augmented_user_review.json")
