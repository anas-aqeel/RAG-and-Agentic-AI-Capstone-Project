"""
Module 3, Exercise 3: Build a Chatbot Interface for the Recommendation System

A Gradio chatbot wrapping the Module 3, Ex 2 multi-agent workflow with:
  - Natural-language intent classification (restaurant / recipe / both / clarification / database)
  - Preference extraction from free-form user input
  - Multi-tab UI: Chat, Add Restaurant, Add Recipe, About

Uses Claude via Google Vertex AI as the LLM backend.

Usage:
    1. Create .env with GCP_PROJECT_ID and GCP_REGION
    2. gcloud auth application-default login
    3. pip install -r requirements.txt
    4. python exercise_3.py        # launches Gradio at http://127.0.0.1:7860
"""

import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr
from anthropic import AnthropicVertex
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Vertex AI client (lazy init)
# =============================================================================

_vertex_client = None


def _get_client() -> AnthropicVertex:
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = AnthropicVertex(
            project_id=os.getenv("GCP_PROJECT_ID"),
            region=os.getenv("GCP_REGION", "us-east5"),
        )
    return _vertex_client


CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5@20251001")  # Haiku is fast for chat


def llm_invoke(system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


# =============================================================================
# Intent classification
# =============================================================================

VALID_INTENTS = ["restaurant", "recipe", "both", "clarification", "database"]


def classify_intent(user_message: str) -> str:
    system_prompt = """You are an intent classifier for a food recommendation system.

Analyze the user's message and classify it as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?" -> restaurant
"How do I make lasagna?" -> recipe
"I want dinner ideas" -> both
"What can you help me with?" -> clarification
"I want to add a new restaurant" -> database

Respond with ONLY the classification label."""

    intent = llm_invoke(system_prompt, user_message, max_tokens=20).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "clarification"
    return intent


# =============================================================================
# Preference extraction
# =============================================================================

def extract_preferences(user_message: str) -> Dict[str, Any]:
    system_prompt = """You are a preference extractor for a food recommendation system.

Extract user preferences from their message and return JSON with these keys:
- favorite_cuisines: List of mentioned cuisines (e.g., ["Italian", "Thai"])
- dietary_restrictions: List of dietary needs (e.g., ["vegetarian", "gluten-free"])
- dining_occasion: Type of dining (e.g., "casual", "fine dining", "quick bite")
- price_range: Price preference (e.g., "$", "$$", "$$$", "$$$$")
- flavor_preferences: List of flavor preferences (e.g., ["spicy", "sweet"])
- other_preferences: Any other relevant details

If a field is not mentioned, use an empty list or "not specified".

Respond with ONLY valid JSON, no markdown fences."""

    response = llm_invoke(system_prompt, user_message, max_tokens=512)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": "",
        }


# =============================================================================
# Mock multi-agent workflow (placeholder for Module 3 Ex 2 integration)
# =============================================================================

def run_recommendation_workflow(preferences: Dict[str, Any], recommendation_type: str) -> Dict[str, Any]:
    """In production this would call the Ex 2 multi-agent workflow."""
    print(f"Running workflow for {recommendation_type} recommendations...")

    mock = {
        "restaurants": [
            {
                "name": "Green Leaf Bistro",
                "cuisine": "Mediterranean",
                "price": "$$",
                "reasoning": "Aligns with healthy, plant-based options and offers diverse Mediterranean menu with vegetarian choices.",
            },
            {
                "name": "Spice Route",
                "cuisine": "Indian",
                "price": "$$",
                "reasoning": "Authentic Indian cuisine with extensive vegetarian options. Spice level customizable to your preference.",
            },
        ],
        "recipes": [
            {
                "name": "One-Pot Chickpea Curry",
                "cuisine": "Indian",
                "difficulty": "Easy",
                "reasoning": "Flavorful, protein-rich dish matching bold flavor preference. Ready in 30 minutes.",
            },
            {
                "name": "Mediterranean Quinoa Bowl",
                "cuisine": "Mediterranean",
                "difficulty": "Easy",
                "reasoning": "Nutritious and satisfying — Mediterranean flavors with plant-based protein.",
            },
        ],
    }

    if recommendation_type == "restaurant":
        return {"restaurants": mock["restaurants"]}
    if recommendation_type == "recipe":
        return {"recipes": mock["recipes"]}
    return mock


# =============================================================================
# Recommendation formatter
# =============================================================================

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    output = ""

    if recommendations.get("restaurants"):
        output += "**Restaurant Recommendations:**\n\n"
        for i, r in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {r['name']}**\n"
            output += f"   - Cuisine: {r['cuisine']}\n"
            output += f"   - Price: {r['price']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    if recommendations.get("recipes"):
        output += "**Recipe Recommendations:**\n\n"
        for i, r in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {r['name']}**\n"
            output += f"   - Cuisine: {r['cuisine']}\n"
            output += f"   - Difficulty: {r['difficulty']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    return output or "I couldn't generate recommendations. Please try again with more details."


# =============================================================================
# Main chatbot function
# =============================================================================

def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        intent = classify_intent(message)
        print(f"Classified intent: {intent}")

        if intent == "clarification":
            return (
                "I'm your food recommendation assistant! I can help you with:\n\n"
                "- **Restaurant recommendations** - cuisine, dietary restrictions, occasion\n"
                "- **Recipe recommendations** - what you'd like to cook\n"
                "- **Database management** - add, update, delete restaurants and recipes\n\n"
                "Just describe what you're looking for!"
            )

        if intent == "database":
            return (
                "To manage the database, please use the tabs above:\n\n"
                "- **Add Restaurant**: submit a new restaurant\n"
                "- **Add Recipe**: submit a new recipe\n\n"
                "Anything else I can help with?"
            )

        if intent in ("restaurant", "recipe", "both"):
            preferences = extract_preferences(message)
            print(f"Extracted preferences: {preferences}")
            recommendations = run_recommendation_workflow(preferences, intent)
            return format_recommendations(recommendations)

        return "I'm not sure how to help with that. Can you rephrase your request?"

    except Exception as e:
        return f"I encountered an error: {e}. Make sure your Vertex AI credentials are configured."


# =============================================================================
# Database management (mock — would write to ChromaDB in production)
# =============================================================================

def add_restaurant(name: str, cuisine: str, price: str, location: str, description: str) -> str:
    print(f"Adding restaurant: {name}")
    return f"Successfully added '{name}' to the database."


def add_recipe(name: str, cuisine: str, difficulty: str, prep_time: str, ingredients: str, instructions: str) -> str:
    print(f"Adding recipe: {name}")
    return f"Successfully added '{name}' recipe to the database."


# =============================================================================
# Build Gradio interface
# =============================================================================

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Food Recommendation Chatbot\n\n"
            "Your personal AI assistant for restaurant and recipe recommendations."
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                gr.ChatInterface(
                    fn=recommendation_chatbot,
                    examples=[
                        "I'm looking for vegetarian restaurants",
                        "Suggest some easy recipes for dinner",
                        "I want spicy Thai food recommendations",
                        "What can you help me with?",
                    ],
                    title="Chat with the Recommendation Assistant",
                    description="Describe your food preferences and I'll recommend restaurants or recipes.",
                )

            with gr.Tab("Add Restaurant"):
                gr.Markdown("### Add a New Restaurant to the Database")
                with gr.Row():
                    with gr.Column():
                        rest_name = gr.Textbox(label="Restaurant Name")
                        rest_cuisine = gr.Textbox(label="Cuisine Type")
                        rest_price = gr.Dropdown(choices=["$", "$$", "$$$", "$$$$"], label="Price Range")
                    with gr.Column():
                        rest_location = gr.Textbox(label="Location")
                        rest_description = gr.Textbox(label="Description", lines=3)

                add_rest_btn = gr.Button("Add Restaurant", variant="primary")
                rest_output = gr.Textbox(label="Status")
                add_rest_btn.click(
                    fn=add_restaurant,
                    inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                    outputs=rest_output,
                )

            with gr.Tab("Add Recipe"):
                gr.Markdown("### Add a New Recipe to the Database")
                with gr.Row():
                    with gr.Column():
                        recipe_name = gr.Textbox(label="Recipe Name")
                        recipe_cuisine = gr.Textbox(label="Cuisine Type")
                        recipe_difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Difficulty")
                    with gr.Column():
                        recipe_time = gr.Textbox(label="Prep Time")
                        recipe_ingredients = gr.Textbox(label="Ingredients (comma-separated)", lines=3)

                recipe_instructions = gr.Textbox(label="Instructions", lines=5)
                add_recipe_btn = gr.Button("Add Recipe", variant="primary")
                recipe_output = gr.Textbox(label="Status")
                add_recipe_btn.click(
                    fn=add_recipe,
                    inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                    outputs=recipe_output,
                )

            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About This Chatbot

                    This chatbot uses a multi-agent AI system (Claude via Vertex AI)
                    to provide personalized food recommendations.

                    ### Features
                    - Intent classification (restaurant / recipe / both / clarification / database)
                    - Preference extraction from natural language
                    - Multi-agent workflow integration (from Module 3 Ex 2)
                    - Database management forms

                    ### Stack
                    - Claude (Sonnet/Haiku) via Google Vertex AI
                    - Gradio for the user interface
                    - ChromaDB for vector retrieval (Module 2)
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
