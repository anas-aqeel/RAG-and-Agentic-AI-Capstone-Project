"""
Module 1, Exercise 1: Structure Unstructured Restaurant Data with an LLM
Standalone script using Claude via Google Vertex AI

Usage:
    1. pip install -r requirements.txt
    2. gcloud auth application-default login
    3. Copy .env.example to .env and fill in GCP_PROJECT_ID
    4. python exercise_1.py
"""

import os
import json
import urllib.request
from dotenv import load_dotenv
from anthropic import AnthropicVertex
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

load_dotenv(dotenv_path="../../.env")


# =============================================================================
# Step 1: Load the Data
# =============================================================================

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"
DATA_FILE = "data/California-Culinary-Map.txt"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"Downloaded {DATA_FILE}")

with open(DATA_FILE, "r") as f:
    restaurant_data = f.read()

print(restaurant_data[:100])

restaurant_list = restaurant_data.split("\n\n")[1:]
print(f"Number of restaurants: {len(restaurant_list)}")
print(restaurant_list[0])


# =============================================================================
# Step 2: Define the LLM
# =============================================================================

vertex_client = AnthropicVertex(
    project_id=os.getenv("GCP_PROJECT_ID"),
    region=os.getenv("GCP_REGION", "us-east5"),
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5@20251001")


def llm_model(system_msg, prompt_txt):
    message = vertex_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_msg,
        messages=[{"role": "user", "content": prompt_txt}],
    )
    return message.content[0].text


print("\n=== Testing LLM ===")
print(llm_model("You are a helpful assistant.", "Which place is warmer in winter? Hawaii or Greenland?"))


# =============================================================================
# Step 3: Prompt Engineering
# =============================================================================

EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1]
EXAMPLE_OUTPUT = """{
  "name": "Mar de Cortez",
  "location": "Santa Monica",
  "type": "casual taqueria",
  "food_style": "Baja-style seafood",
  "rating": 4.2,
  "price_range": 1,
  "signatures": ["shrimp tacos with mango-habanero salsa", "grilled fish burritos"],
  "vibe": "salt-air energy",
  "environment": "a premier sun-drenched spot for open-air dining near the pier.",
  "shortcomings": []
}"""


def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = (
        "You are a data extraction assistant. Extract structured information from "
        "restaurant descriptions and output valid JSON. Follow the exact schema "
        "provided. For price_range, convert dollar signs ($$, $$$) into an integer. "
        "Output ONLY the JSON object, nothing else."
    )
    base_user_prompt = f"""Task:
Extract structured restaurant data as a valid JSON object.

Restaurant description:
{restaurant_paragraph}

Example:
Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
Output:
{EXAMPLE_OUTPUT}
"""
    return base_system_msg, base_user_prompt


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


# Test prompt on first restaurant
print("\n=== Testing Prompt ===")
sys_msg, user_prompt = restaurant_data_structure_prompt_generation(restaurant_list[0])
test_response = llm_model(system_msg=sys_msg, prompt_txt=user_prompt)
print(test_response)

try:
    obj = Restaurant.model_validate_json(test_response)
    print(f"Validation SUCCESS: {obj.name}")
except ValidationError as e:
    print(f"Validation FAILED: {e.json()}")


# =============================================================================
# Step 4: Structure All Restaurant Data
# =============================================================================


def JSON_auto_repair_prompts(candidate_json_output, error_message):
    repair_system_msg = (
        "You are a JSON repair expert. Fix invalid JSON to match the required schema. "
        "Output ONLY the corrected JSON object, nothing else."
    )
    repair_prompt = f"""Fix this invalid JSON based on the error message.

Invalid JSON:
{candidate_json_output}

Error:
{error_message}

Output ONLY the corrected JSON."""
    return repair_system_msg, repair_prompt


print("\n=== Processing All Restaurants ===")
error_log = []
structured_restaurant_lists = []

for i, paragraph in enumerate(restaurant_list):
    try:
        sys_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
        candidate = llm_model(system_msg=sys_msg, prompt_txt=user_prompt)

        # FIX: Use a for-loop so the last repaired candidate is also validated.
        # Original code used a while loop that exited before validating the
        # 3rd repair, silently discarding a potentially valid result.
        valid = False
        for retries in range(4):  # 1 initial attempt + up to 3 repairs
            try:
                Restaurant.model_validate_json(candidate)
                valid = True
                break
            except ValidationError as e:
                if retries < 3:
                    error_log.append(f"[{i}] Validation retry {retries}: {e}")
                    repair_sys, repair_prompt = JSON_auto_repair_prompts(candidate, e.json())
                    candidate = llm_model(system_msg=repair_sys, prompt_txt=repair_prompt)

        if not valid:
            error_log.append(f"[{i}] SKIPPED: max retries reached")
            continue

        structured_restaurant_lists.append(candidate)

    except Exception as e:
        error_log.append(f"[{i}] CRASHED: {e}")
        continue

    if (i + 1) % 20 == 0:
        print(f"{i + 1} out of {len(restaurant_list)} is done")

print(f"ALL DONE!! Processed: {len(structured_restaurant_lists)}, Errors: {len(error_log)}")

# Print 50th item
if len(structured_restaurant_lists) >= 50:
    print(f"\n=== 50th Restaurant ===")
    print(structured_restaurant_lists[49])
else:
    print(f"\nOnly {len(structured_restaurant_lists)} restaurants processed; cannot show 50th.")

# Save to JSON
structured_json = [json.loads(r) for r in structured_restaurant_lists]
for i, item in enumerate(structured_json):
    item["itemId"] = 1000001 + i

with open("data/structured_restaurant_data.json", "w", encoding="utf-8") as f:
    json.dump(structured_json, f, indent=4)

print(f"\nSaved {len(structured_json)} restaurants to data/structured_restaurant_data.json")