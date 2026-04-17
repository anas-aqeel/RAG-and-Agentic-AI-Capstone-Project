"""
Module 1, Exercise 3: Build a Command-Line Data Management UI
IBM watsonx Submission — restaurant_data_management.py

Copy this entire file content into the IBM lab's restaurant_data_management.py
"""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = "structured_restaurant_data.json"
BACKUP_PATH = "structured_restaurant_data.json.bak"

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** "
    "rating, it captures the salt-air energy of the coast through its signature "
    "beer-battered snapper tacos and zesty octopus ceviche, making it a premier "
    "spot for open-air dining near the pier. Price range: $"
)

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


# =============================================================================
# Helper functions (provided by lab)
# =============================================================================


def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    """Displays restaurant data in a clean, vertical format."""
    print(f"\n{'='*15} RESTAURANT #{index} {'='*15}")
    name = res.get("name", res.get("restaurant_name", "Unnamed Restaurant"))
    print(f"NAME : {name}")

    for key, value in res.items():
        if key.lower() not in ["name", "restaurant_name"]:
            label = key.replace("_", " ").upper()
            print(f"{label:<12}: {value}")
    print("=" * 45)


class Restaurant(BaseModel):
    """The restaurant pydantic schema used in lesson 1."""

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


# =============================================================================
# Exercise 1: Integrate LLM from Lesson 1 (IBM watsonx version)
# =============================================================================


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


def llm_model(system_msg, prompt_txt, params=None):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"

    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt},
    ]

    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]


def JSON_auto_repair_prompts(response, error_message):
    repair_system_msg = (
        "You are a JSON repair expert. Fix invalid JSON to match the required schema. "
        "Output ONLY the corrected JSON object, nothing else."
    )
    repair_prompt = f"""Fix this invalid JSON based on the error message.

Invalid JSON:
{response}

Error:
{error_message}

Output ONLY the corrected JSON."""
    return repair_system_msg, repair_prompt


def new_data_entry_process(paragraph, itemId):
    """Process a new restaurant paragraph into structured JSON with auto-repair."""
    sys_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate = llm_model(system_msg=sys_msg, prompt_txt=user_prompt)

    valid = False
    for retries in range(4):  # 1 initial + up to 3 repairs
        try:
            Restaurant.model_validate_json(candidate)
            valid = True
            break
        except ValidationError as e:
            if retries < 3:
                repair_sys, repair_prompt = JSON_auto_repair_prompts(
                    candidate, e.json()
                )
                candidate = llm_model(system_msg=repair_sys, prompt_txt=repair_prompt)

    if not valid:
        print("Failed to generate valid restaurant data after retries.")
        return None

    new_entry = json.loads(candidate)
    new_entry["itemId"] = itemId
    return new_entry


# =============================================================================
# Exercise 2: The main UI function
# =============================================================================


def manage_restaurants(file_path=FILEPATH, backup_path=BACKUP_PATH):
    while True:
        data = load_data(file_path)
        print(f"\n\U0001f3e8 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == "1":
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                name = res.get("name", "N/A")
                print(f"  [{i}] {name}")

        elif choice == "2":
            idx_input = input("Enter record index: ")
            try:
                idx = int(idx_input)
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid index.")

        elif choice in ["3", "4", "5"]:
            print("\n\u2757 SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            if choice == "3":  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter the new restaurant description:\n")
                new_entry = new_data_entry_process(paragraph, itemId)
                if new_entry:
                    data.append(new_entry)
                    save_data(data, file_path, backup_path)
                    print("\u2705 Restaurant added.")

            elif choice == "4":  # EDIT DATA
                idx_input = input("Enter record index to edit: ")
                try:
                    idx = int(idx_input)
                    if 0 <= idx < len(data):
                        record = data[idx]
                        for key in list(record.keys()):
                            current = record[key]
                            new_val = input(
                                f"  {key} [{current}] (Enter to skip): "
                            )
                            if new_val.strip():
                                record[key] = new_val
                        save_data(data, file_path, backup_path)
                        print("\u2705 Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

            elif choice == "5":  # DELETE DATA
                idx_input = input("Enter record index to delete: ")
                try:
                    idx = int(idx_input)
                    if 0 <= idx < len(data):
                        data.pop(idx)
                        save_data(data, file_path, backup_path)
                        print("\u2705 Restaurant deleted.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

        elif choice == "6":  # EXIT
            break
        else:
            print("Invalid input.")


# =============================================================================
# Exercise 3: Unit Tests
# =============================================================================


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = "structured_restaurant_data_unit_test.json"
        self.test_file_backup = "structured_restaurant_data_unit_test.json.bak"
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, "w") as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant, then delete it.
        """
        mock_restaurant = (
            "The Copper Sprout is a high-concept, Modern Appalachian farm-to-table "
            "destination that blends an industrial-chic aesthetic with rustic forest "
            "charm, featuring reclaimed wood and amber lighting to create a sophisticated "
            "yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal "
            "foraging and local heritage, headlined by signature dishes like Cast-Iron "
            "Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom "
            "Risotto with aged goat cheese. The experience is designed to be intimate "
            "and earthy, making it a premier spot for those seeking high-quality, "
            "smokehouse-influenced cuisine in a refined, atmospheric setting."
        )
        mock_input.side_effect = ["3", "yes", mock_restaurant, "6"]

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("\u2705 Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ["5", "yes", "1", "6"]

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        """
        mock_input.side_effect = ["5", "no", "6"]

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call
