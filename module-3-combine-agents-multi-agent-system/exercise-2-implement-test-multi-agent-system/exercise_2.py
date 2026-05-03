"""
Module 3, Exercise 2: Implement and Test a Multi-Agent Recommendation System

Builds a hybrid multi-agent workflow with 4 phases:
  Phase 1 (sequential): User Profile Generator
  Phase 2 (sequential): RAG Retriever
  Phase 3 (parallel):   Food Trend Analyst, Food Style Expert, Nutrition Expert
  Phase 4 (sequential): Recommendation Expert

Uses Claude via Google Vertex AI as the LLM backend.

Usage:
    1. Create .env with GCP_PROJECT_ID and GCP_REGION
    2. gcloud auth application-default login
    3. pip install -r requirements.txt
    4. python exercise_2.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

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


CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6@20260401")


# =============================================================================
# Agent configurations (from Exercise 1)
# =============================================================================

agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": (
            "You are an expert user behavior analyst with 10 years of experience in the food industry. "
            "You excel at identifying patterns in dining behavior and building rich user profiles."
        ),
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": "You are a data retrieval specialist with expertise in vector databases and semantic search.",
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": "You are a culinary journalist who has spent 15 years covering food culture across global markets.",
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": "You are a trained chef and culinary anthropologist with expertise in global cuisines.",
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": "You are a registered dietitian with 8 years of clinical experience.",
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": "You are a recommendation systems architect with experience in personalization engines.",
    },
}


# =============================================================================
# Helper: call_agent
# =============================================================================

def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent and return its response. Always asks for raw JSON."""
    config = agent_configs[agent_key]

    system_prompt = (
        f"You are a {config['role']}.\n\n"
        f"Your goal: {config['goal']}\n\n"
        f"Your background: {config['backstory']}\n\n"
        "Respond with valid JSON only — no markdown fences, no preamble."
    )

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text.strip()

    # Strip markdown code fences if model added them
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


# =============================================================================
# Node functions — each reads state, runs an agent, writes back
# =============================================================================

def node_generate_profile(state: dict) -> dict:
    print("\n[Phase 1] Generating user profile...")
    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"  User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"  Error: {e}")
        user_profile = {"error": str(e)}

    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state


def node_retrieve_candidates(state: dict) -> dict:
    print("\n[Phase 2] Retrieving candidates...")
    user_message = f"""Based on this user profile:
{json.dumps(state['user_profile'], indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    try:
        response = call_agent("rag_retriever", user_message)
        retrieved = json.loads(response)
        restaurants = retrieved.get("restaurants", [])
        recipes = retrieved.get("recipes", [])
        print(f"  Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"  Error: {e}")
        restaurants, recipes = [], []

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state


def node_analyze_trends(state: dict) -> dict:
    print("\n[Phase 3a] Analyzing trends...")
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"  Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"  Error: {e}")
        trend_analysis = {"error": str(e)}

    state["trend_analysis"] = trend_analysis
    return state


def node_analyze_styles(state: dict) -> dict:
    print("\n[Phase 3b] Analyzing food styles...")

    user_message = f"""Analyze the cuisine types and flavor profiles in these options against the user's preferences:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}

Identify the dominant cuisines, flavor profiles (umami, spicy, sweet, sour, etc.),
and explain how well each matches the user's stated preferences.

Return JSON:
{{
  "dominant_cuisines": [str],
  "flavor_profiles": [{{"profile": str, "examples": [str]}}],
  "style_matches": [{{"item": str, "match_score": float, "reasoning": str}}]
}}
"""
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"  Style analysis complete")
    except Exception as e:
        print(f"  Error: {e}")
        style_analysis = {"error": str(e)}

    state["style_analysis"] = style_analysis
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    print("\n[Phase 3c] Evaluating nutrition...")
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"  Nutrition evaluation complete")
    except Exception as e:
        print(f"  Error: {e}")
        nutrition_analysis = {"error": str(e)}

    state["nutrition_analysis"] = nutrition_analysis
    return state


def node_generate_recommendations(state: dict) -> dict:
    print("\n[Phase 4] Generating final recommendations...")
    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"  {len(recommendations.get('restaurants', []))} restaurants, "
              f"{len(recommendations.get('recipes', []))} recipes recommended")
    except Exception as e:
        print(f"  Error: {e}")
        recommendations = {"error": str(e)}

    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state


# =============================================================================
# Workflow orchestrator (hybrid: sequential + parallel)
# =============================================================================

def run_workflow(user_input: str) -> dict:
    state = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }

    # Phase 1
    state = node_generate_profile(state)

    # Phase 2
    state = node_retrieve_candidates(state)

    # Phase 3 — parallel
    print("\n[Phase 3] Running analysis agents in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_trends = executor.submit(node_analyze_trends, dict(state))
        f_styles = executor.submit(node_analyze_styles, dict(state))
        f_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        r_trends = f_trends.result()
        r_styles = f_styles.result()
        r_nutrition = f_nutrition.result()

    state["trend_analysis"] = r_trends["trend_analysis"]
    state["style_analysis"] = r_styles["style_analysis"]
    state["nutrition_analysis"] = r_nutrition["nutrition_analysis"]

    # Phase 4
    state = node_generate_recommendations(state)
    return state


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_recommendations(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("RECOMMENDATION EVALUATION")
    print("=" * 80)

    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])

    print(f"\nRestaurant recommendations: {len(restaurants)}")
    print(f"Recipe recommendations:     {len(recipes)}")

    diet = profile.get("dietary_restrictions", [])
    if diet:
        print(f"\nDietary restrictions: {', '.join(diet)}")

    cuisines = profile.get("favorite_cuisines", [])
    if cuisines:
        print(f"Favorite cuisines:    {', '.join(cuisines)}")

    if restaurants:
        r = restaurants[0]
        print(f"\nFirst restaurant: {r.get('name', 'N/A')}")
        print(f"  Reasoning: {r.get('reasoning', 'N/A')}")

    if recipes:
        r = recipes[0]
        print(f"\nFirst recipe: {r.get('name', 'N/A')}")
        print(f"  Reasoning: {r.get('reasoning', 'N/A')}")
    print("=" * 80)


# =============================================================================
# Test cases
# =============================================================================

TEST_USER_HEALTH = """
Restaurant Visit History:
- Visited "Green Bowl" (Vegan, $$) 8 times
- Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
- Visited "Juice Lab" (Smoothies, $) 3 times

Social Media Posts:
- "Loving my plant-based journey!"
- "This gluten-free Mediterranean bowl is amazing!"
- "Fresh juice is the best way to start the day."

Dietary Restrictions: Vegan, Gluten-Free
"""

TEST_USER_FOODIE = """
Restaurant Visit History:
- Visited "Omakase Sushi" (Japanese Fine Dining, $$$$) 4 times
- Visited "Street Food Market" (International Fusion, $$) 6 times
- Visited "Molecular Gastronomy Lab" (Experimental, $$$$) 2 times

Social Media Posts:
- "Mind-blown by the 12-course tasting menu!"
- "Trying crickets for the first time. Surprisingly good!"
- "This molecular take on traditional ramen is art."

Dietary Restrictions: None
"""


if __name__ == "__main__":
    print("=" * 80)
    print("TEST CASE 1: Health-Conscious User")
    print("=" * 80)
    result_1 = run_workflow(TEST_USER_HEALTH)
    print("\nFINAL RECOMMENDATIONS:")
    print(json.dumps(result_1["final_recommendations"], indent=2))
    evaluate_recommendations(result_1)

    print("\n\n" + "=" * 80)
    print("TEST CASE 2: Adventurous Foodie")
    print("=" * 80)
    result_2 = run_workflow(TEST_USER_FOODIE)
    print("\nFINAL RECOMMENDATIONS:")
    print(json.dumps(result_2["final_recommendations"], indent=2))
    evaluate_recommendations(result_2)
