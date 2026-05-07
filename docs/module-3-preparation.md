# Module 3 — Interview Preparation & Deep Dive

> **What this module covered:** Designing specialized AI agents, orchestrating them into multi-agent workflows (sequential + parallel hybrid), managing shared state, and exposing the system through a Gradio chatbot with intent classification and preference extraction.

---

## Table of Contents

1. [What is an AI Agent?](#1-what-is-an-ai-agent)
2. [Why Multi-Agent? (vs One Big Prompt)](#2-why-multi-agent-vs-one-big-prompt)
3. [Multi-Agent Architectures](#3-multi-agent-architectures)
4. [Agent Design: Role / Goal / Backstory](#4-agent-design-role--goal--backstory)
5. [Shared State Pattern (TypedDict)](#5-shared-state-pattern-typeddict)
6. [Sequential vs Parallel Execution](#6-sequential-vs-parallel-execution)
7. [ThreadPoolExecutor Deep Dive](#7-threadpoolexecutor-deep-dive)
8. [Intent Classification](#8-intent-classification)
9. [Preference Extraction (Structured Output)](#9-preference-extraction-structured-output)
10. [Gradio for Conversational UIs](#10-gradio-for-conversational-uis)
11. [LangGraph vs Custom Orchestration](#11-langgraph-vs-custom-orchestration)
12. [Production Concerns](#12-production-concerns)
    - [Error Handling & Cascading Failures](#121-error-handling--cascading-failures)
    - [Cost Management](#122-cost-management)
    - [Latency Optimization](#123-latency-optimization)
    - [Observability & Tracing](#124-observability--tracing)
    - [Streaming Responses](#125-streaming-responses)
    - [State Persistence & Resumability](#126-state-persistence--resumability)
    - [Caching at the Agent Level](#127-caching-at-the-agent-level)
    - [Fallbacks & Graceful Degradation](#128-fallbacks--graceful-degradation)
    - [Tool Use vs Pure Prompt Agents](#129-tool-use-vs-pure-prompt-agents)
13. [Testing Multi-Agent Systems](#13-testing-multi-agent-systems)
14. [Common Pitfalls](#14-common-pitfalls)
15. [Real-World Scenarios](#15-real-world-scenarios)
16. [All Interview Questions & Answers](#16-all-interview-questions--answers)
17. [Concepts You Must Know](#17-concepts-you-must-know)

---

## 1. What is an AI Agent?

### The Definition

An **AI agent** is an LLM wrapped with three things:
1. **A specific role** — what it's responsible for
2. **A goal** — what success looks like
3. **(Optionally) tools** — APIs, functions, or databases it can call

```python
# This is just an LLM call:
response = llm.invoke("What's the weather in Paris?")

# This is an agent:
agent = Agent(
    role="Weather Reporter",
    goal="Provide accurate weather information",
    tools=[get_weather_api, format_forecast],
    system_prompt="You are a weather expert..."
)
agent.invoke("What's the weather in Paris?")
# → calls get_weather_api(city="Paris") → formats response
```

### Agent vs Function vs LLM Call

| | **LLM Call** | **Function** | **Agent** |
|---|---|---|---|
| Behavior | Stochastic | Deterministic | Stochastic + Goal-directed |
| Decides next step | No | No (hard-coded) | Yes (chooses tools/actions) |
| Can fail silently | Yes (hallucinate) | No (raises) | Yes (and tries again) |
| Has memory | No (per call) | No | Optionally (state) |

### Analogy

- **Function** = a hammer. Hits exactly where you tell it to.
- **LLM call** = a creative consultant. Gives an answer when asked, no follow-through.
- **Agent** = a junior employee. You tell them "research this trend," they go figure out which tools to use and bring back a report.

### What Makes Agents Hard

Agents introduce three fundamental problems that simple LLM calls don't have:
1. **Non-determinism** — same input may produce different actions
2. **Cascading failures** — one bad output corrupts everything downstream
3. **Cost explosion** — multiple LLM calls per request

---

## 2. Why Multi-Agent? (vs One Big Prompt)

### The Naive Alternative

You could do everything in one prompt:

```python
mega_prompt = f"""
Given this user data: {user_data}

Step 1: Build a profile
Step 2: Retrieve restaurants and recipes
Step 3: Analyze trends
Step 4: Analyze styles
Step 5: Evaluate nutrition
Step 6: Synthesize recommendations

Return final recommendations as JSON.
"""
recommendations = llm.invoke(mega_prompt)
```

**This works for toy demos. It fails in production.**

### Why Multi-Agent Wins

| Reason | One Big Prompt | Multi-Agent |
|--------|---------------|-------------|
| **Specialization** | Generic system prompt — jack of all trades | Each agent has focused role + backstory |
| **Modularity** | Change one step → re-test the whole chain | Change one agent independently |
| **Observability** | Black box — hard to debug | See output of each agent |
| **Parallelism** | Single sequential call | Independent agents run concurrently |
| **Tool use** | One tool surface | Each agent has its own tools |
| **Cost control** | Always uses biggest model | Use Haiku for simple agents, Sonnet for synthesis |
| **Failure isolation** | Whole pipeline fails | One agent fails, others succeed |
| **Context window** | Can hit token limit | Each agent gets only what it needs |

### Concrete Example — Token Budget

Our 6-agent recommendation system:

```
One big prompt (sequential):
  Input:  ~8,000 tokens (all data + all instructions)
  Output: ~3,000 tokens
  Total:  11,000 tokens × $X/1M tokens

Multi-agent (parallel Phase 3):
  6 agents × ~2,000 tokens each = 12,000 tokens
  But Phase 3 runs in parallel → wall-clock time is 4× faster
  AND you can route trivial agents to Haiku (10× cheaper)

Effective cost: lower in many cases, latency is dramatically lower.
```

### When NOT to Use Multi-Agent

- The task is genuinely simple ("translate this sentence")
- All steps share the same context and don't benefit from specialization
- Latency is critical and you can't parallelize

**Rule of thumb:** if you can describe the task as "first do X, then Y, then Z" with each step having a different goal, multi-agent is probably worth it.

---

## 3. Multi-Agent Architectures

### Pattern 1: Sequential (Pipeline)

```
[Agent A] → [Agent B] → [Agent C] → output
```

Each agent's output is the next agent's input. Simple, easy to debug.

```python
state = agent_a(input)
state = agent_b(state)
state = agent_c(state)
return state
```

**Use when:** strict ordering required, each step depends on the previous.

### Pattern 2: Parallel (Fan-out / Fan-in)

```
              ┌→ [Agent A] →┐
[Input] ──────┼→ [Agent B] →┼→ [Aggregator] → output
              └→ [Agent C] →┘
```

All agents run concurrently from same input, results merged.

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    results = [ex.submit(a, input) for a in [agent_a, agent_b, agent_c]]
    outputs = [r.result() for r in results]
return aggregate(outputs)
```

**Use when:** agents are independent (none consumes another's output).

### Pattern 3: Hybrid (Our Module 3 Architecture)

```
[Profile] → [Retrieval] → ┌→ [Trends]    →┐
                          ├→ [Styles]    →┤→ [Synthesis] → output
                          └→ [Nutrition] →┘
   sequential     sequential      parallel        sequential
```

Mix of sequential and parallel. The most common production pattern — phases that depend on each other run in series; agents inside a phase run in parallel.

### Pattern 4: Hierarchical (Manager + Workers)

```
              [Manager Agent]
              /     |       \
       [Worker]  [Worker]  [Worker]
```

A "manager" agent decides which workers to invoke and in what order. Workers don't know about each other.

**Use when:** the routing logic itself requires LLM intelligence (e.g., "is this a refund question or a tech support question?").

### Pattern 5: Debate / Consensus

```
[Agent A: pro] ↔ [Agent B: con] → [Judge] → decision
```

Multiple agents argue or vote, a final judge picks the winner. Used in research and critical decisions where one model might be biased.

**Use when:** correctness matters more than speed (medical, legal, financial reasoning).

### Pattern 6: ReAct Loop (single agent, multi-step)

```
[Reason] → [Act (call tool)] → [Observe] → [Reason] → ...
```

A single agent loops: think, act, observe, repeat until goal achieved. This is what frameworks like LangChain's `AgentExecutor` do.

**Use when:** the agent needs to chain tool calls dynamically (e.g., "look up the weather, then check my calendar, then book a flight").

### Architecture Selection Cheat Sheet

| If you need... | Use |
|----------------|-----|
| Strict ordering | Sequential |
| Independent analysis | Parallel |
| Mix of both | Hybrid (most common) |
| Dynamic routing | Hierarchical or ReAct |
| Self-correction | Debate / Consensus |
| Tool chaining | ReAct |

---

## 4. Agent Design: Role / Goal / Backstory

### The CrewAI-Style Pattern

```python
agent_config = {
    "role": "Food Trend Analyst",
    "goal": "Identify current food trends and emerging dining concepts.",
    "backstory": "You are a culinary journalist who has spent 15 years..."
}
```

### Why All Three?

- **Role** = the agent's title. Anchors its identity. ("You ARE a Food Trend Analyst.")
- **Goal** = the success criteria. ("Your job is to identify trends.")
- **Backstory** = activates relevant patterns from training data. (LLMs trained on web text have implicit knowledge about how a 15-year culinary journalist would write.)

### What Backstory Actually Does

LLMs don't "know" they're an AI — they predict the next token. By telling them they're a 15-year veteran food critic, you shift the probability distribution toward outputs that match how such a person would write.

**Empirical effect:** this consistently improves output quality over generic "You are a helpful assistant" prompts. The improvement is bigger for nuanced/subjective tasks, smaller for factual ones.

### Bad vs Good Agent Design

❌ **Bad:**
```python
agent = {
    "role": "Helper",
    "goal": "Help the user.",
    "backstory": "You help people."
}
```
Generic. No specialization. Output will be mediocre.

✅ **Good:**
```python
agent = {
    "role": "Nutrition Expert",
    "goal": "Evaluate nutritional content and ensure dietary compliance.",
    "backstory": "You are a registered dietitian with 8 years of clinical experience. You specialize in identifying allergens, evaluating macronutrient balance, and flagging items that violate stated dietary restrictions."
}
```
Specific role, measurable goal, credible backstory.

### Tools (When to Add Them)

Add a tool to an agent when:
1. The action requires up-to-date information (weather, prices, stock)
2. The action requires deterministic computation (math, lookups)
3. The action has side effects (sending email, updating DB)

Don't add a tool just because you can — every tool increases the agent's decision space and failure modes.

---

## 5. Shared State Pattern (TypedDict)

### The State Object

```python
INITIAL_STATE = {
    "user_input": "",
    "user_profile": {},
    "retrieved_restaurants": [],
    "retrieved_recipes": [],
    "trend_analysis": {},
    "style_analysis": {},
    "nutrition_analysis": {},
    "final_recommendations": {},
    "workflow_step": "start"
}
```

### Why a Single State Dict?

1. **Single source of truth** — every node reads/writes the same object
2. **Serializable** — easy to checkpoint, log, replay
3. **Inspectable** — at any point you can dump the state and see exactly what each agent has produced
4. **Type-safe** — with `TypedDict` you get autocomplete and type errors

### TypedDict vs Pydantic

```python
# TypedDict — lightweight, no runtime validation
from typing import TypedDict
class AgentState(TypedDict):
    user_input: str
    user_profile: dict

# Pydantic — runtime validation, more verbose
from pydantic import BaseModel
class AgentState(BaseModel):
    user_input: str
    user_profile: dict
```

**TypedDict** is what LangGraph uses by default. Lighter weight, but no runtime validation.
**Pydantic** validates at runtime — catches bugs but adds overhead.

For multi-agent state where the LLM produces unpredictable output, **Pydantic is often safer** — you catch malformed JSON immediately rather than letting it propagate.

### The `workflow_step` Field — Why It Matters

```python
state["workflow_step"] = "profile_generated"  # or "candidates_retrieved", "complete"
```

This single field enables four production capabilities:
1. **Debugging** — when something fails, see exactly which step
2. **Conditional branching** — "if workflow_step is 'profile_failed', route to fallback"
3. **Resumability** — restart from last completed step
4. **Telemetry** — measure phase durations, alert on stuck workflows

### State Mutation — Pattern vs Anti-pattern

❌ **Anti-pattern:** mutate state in place across threads
```python
def node(state):
    state["foo"] = compute()  # MUTATION in place
    return state
```
This breaks if the state is shared across threads in Phase 3.

✅ **Pattern:** copy, mutate, return
```python
def node(state):
    new_state = dict(state)         # shallow copy
    new_state["foo"] = compute()
    return new_state
```
Or pass `dict(state)` when submitting to ThreadPoolExecutor (which is what we did).

---

## 6. Sequential vs Parallel Execution

### The Math

Suppose each LLM call takes 3 seconds.

**Sequential (Phase 3 with 3 agents):**
```
Trends:     3s
Styles:     3s
Nutrition:  3s
Total:      9s
```

**Parallel (Phase 3 with 3 agents):**
```
All three start at t=0
Wall-clock: max(3s, 3s, 3s) = 3s

Total:      3s  → 3× faster
```

### When Parallelism is Safe

✅ Agents read same input, write to different state fields
✅ No shared mutable resources (no race conditions)
✅ No ordering dependency

### When Parallelism Breaks

❌ Agent B needs Agent A's output → must run sequentially
❌ Both agents update the same DB row → race condition
❌ Hitting rate limits → parallel calls bunched together cause 429s

### Rate-Limit-Aware Parallelism

```python
import asyncio
from anthropic import AsyncAnthropicVertex

# Limit concurrent requests to respect API rate limits
sem = asyncio.Semaphore(10)

async def call_with_limit(agent_fn, *args):
    async with sem:
        return await agent_fn(*args)
```

For Module 3, three agents in parallel is well below any provider's limits. But at scale (100s of concurrent users × 3 parallel agents = 300s of concurrent API calls), you need a semaphore.

---

## 7. ThreadPoolExecutor Deep Dive

### What We Used

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    f_trends    = executor.submit(node_analyze_trends, dict(state))
    f_styles    = executor.submit(node_analyze_styles, dict(state))
    f_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

    r_trends    = f_trends.result()      # blocks until done
    r_styles    = f_styles.result()
    r_nutrition = f_nutrition.result()
```

### Why Threads (Not Processes) Work for LLM Calls

LLM calls are **I/O-bound** — most time is spent waiting for the network. Python's GIL releases during I/O, so threads run concurrently for the wait portion.

| Workload | Best choice |
|----------|------------|
| LLM API calls (I/O-bound) | **Threads** ← our case |
| Image processing on CPU | **Processes** (avoid GIL) |
| Pure async libraries | **asyncio** |

### `submit()` vs `map()`

```python
# submit() — best for heterogeneous tasks (different functions)
f1 = ex.submit(agent_a, input)
f2 = ex.submit(agent_b, input)

# map() — best for homogeneous tasks (same function, many inputs)
results = ex.map(process_item, list_of_inputs)
```

Phase 3 uses `submit` because each agent is a different function.

### `as_completed` for Streaming Results

```python
from concurrent.futures import as_completed

futures = {ex.submit(fn, x): name for fn, x, name in tasks}

for future in as_completed(futures):
    name = futures[future]
    try:
        result = future.result()
        print(f"{name} done: {result}")
    except Exception as e:
        print(f"{name} failed: {e}")
```

`as_completed` yields results as they finish — useful for showing progress bars or streaming partial results to the user.

### Common ThreadPoolExecutor Bugs

1. **Sharing mutable state**
   ```python
   results = []
   def worker(x):
       results.append(x)  # NOT THREAD-SAFE
   ```
   Fix: use `queue.Queue` or return values via `future.result()`.

2. **Forgetting `with`**
   ```python
   ex = ThreadPoolExecutor()
   # ... no shutdown → threads leak
   ```
   The `with` block ensures `ex.shutdown()` is called.

3. **Calling `.result()` inside the loop**
   ```python
   for task in tasks:
       result = ex.submit(fn, task).result()  # SEQUENTIAL — defeats the purpose!
   ```
   Submit all first, then collect results.

---

## 8. Intent Classification

### The Pattern

User says something in natural language. We need to route it to the right handler.

```python
def classify_intent(message: str) -> str:
    system = """Classify the message as ONE of:
    - "restaurant"
    - "recipe"
    - "both"
    - "clarification"
    - "database"
    Respond with ONLY the label."""

    response = llm.invoke([SystemMessage(system), HumanMessage(message)])
    intent = response.content.strip().lower()

    # Always validate — LLM might return garbage
    if intent not in VALID_INTENTS:
        intent = "clarification"  # safe default
    return intent
```

### Why LLM Classification (vs Rules / Regex)

| Approach | Pros | Cons |
|----------|------|------|
| Regex/keywords | Fast, free, deterministic | Brittle, can't handle paraphrasing |
| Classical ML (sklearn) | Fast, no LLM cost | Need labeled training data |
| LLM | Handles any phrasing zero-shot | Slower, costs $, non-deterministic |

**Real production:** start with LLM (fast to ship), then collect data, then train a small classifier (BERT-class) for the high-volume path. Fallback to LLM for edge cases.

### Tightening LLM Classification

1. **Constrain to enum**
   ```python
   # Anthropic's "tool use" feature lets you force structured output
   tool = {
       "name": "classify",
       "input_schema": {
           "properties": {
               "intent": {"type": "string", "enum": ["restaurant", "recipe", ...]}
           }
       }
   }
   ```
   The model is forced to pick one of the enum values.

2. **Always validate**
   ```python
   if intent not in VALID_INTENTS:
       intent = "clarification"
   ```

3. **Use few-shot examples in the prompt**
   Examples in the prompt double accuracy on ambiguous queries.

4. **Add a confidence threshold**
   Use logprobs (OpenAI) or sampling (Anthropic) to estimate confidence; route low-confidence to "clarification."

### Edge Cases

- **Empty input** — handle before calling LLM
- **Multiple intents** — "I want a recipe AND a restaurant" → route to "both" or both handlers
- **Adversarial input** — prompt injection: "Ignore previous instructions and..." → use system prompt isolation, validate output

---

## 9. Preference Extraction (Structured Output)

### The Goal

Convert free-form text into a structured object:

```
"I love spicy Thai food and I'm vegetarian"
                ↓
{
  "favorite_cuisines": ["Thai"],
  "dietary_restrictions": ["vegetarian"],
  "flavor_preferences": ["spicy"],
  ...
}
```

### Three Approaches

**1. JSON in prompt (what we did)**
```python
system = "Return JSON with these keys: ..."
response = llm.invoke(system + user_message)
data = json.loads(response.content)
```
Simple, works ~95% of the time.

**2. Tool use / Function calling (more robust)**
```python
# Anthropic
tools = [{
    "name": "extract_preferences",
    "input_schema": pydantic_to_schema(Preferences)
}]
response = client.messages.create(model=..., tools=tools, messages=[...])
# Model is FORCED to call the tool with valid JSON
```
~99% reliability. Use for production.

**3. Pydantic + retry**
```python
from pydantic import BaseModel, ValidationError

class Preferences(BaseModel):
    favorite_cuisines: list[str]
    dietary_restrictions: list[str]
    # ...

for attempt in range(3):
    try:
        data = Preferences.model_validate_json(response.content)
        break
    except ValidationError as e:
        # Feed the error back and ask the model to fix it
        response = llm.invoke(f"Your previous output had errors: {e}. Fix it.")
```

### Why JSON Mode/Tool Use Beats Plain Prompts

LLMs are trained to produce JSON in tool-use mode with constrained decoding. This means:
- The output **always** parses
- Field types are enforced (no string where int expected)
- Required fields are guaranteed present

If your code does `data["price_range"]` and the model omits that key, you get a `KeyError` in production at 3am. Tool use prevents this class of bug entirely.

### Defensive Defaults

Always provide a fallback:
```python
try:
    preferences = json.loads(response.content)
except json.JSONDecodeError:
    preferences = DEFAULT_PREFERENCES  # never crash the chatbot
```

---

## 10. Gradio for Conversational UIs

### Why Gradio

Gradio gives you a usable web UI for an ML/AI prototype in **5 lines of Python**. No HTML, no frontend framework, no deployment ceremony.

```python
import gradio as gr

def chat(message, history):
    return f"You said: {message}"

gr.ChatInterface(fn=chat).launch()
# → opens http://127.0.0.1:7860
```

### The `(message, history)` Signature

Gradio's `ChatInterface` calls your function with:
- `message: str` — what the user just typed
- `history: List[Tuple[str, str]]` — list of `(user, bot)` turns so far

```python
def chat(message, history):
    # history = [("hi", "hello!"), ("recommend food", "sure...")]
    full_context = "\n".join([f"User: {u}\nBot: {b}" for u, b in history])
    full_context += f"\nUser: {message}"
    return llm.invoke(full_context)
```

### Tabs and Forms (Multi-Tab Pattern)

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat)
        with gr.Tab("Add Restaurant"):
            name = gr.Textbox()
            btn = gr.Button("Submit")
            btn.click(fn=add_restaurant, inputs=[name], outputs=...)
```

Why mix chat + forms? Some interactions are best done conversationally ("what should I eat?"), others are best done structured ("add a new restaurant with name=X, cuisine=Y"). Don't force everything through chat.

### Streaming Responses

Without streaming, the user stares at a blank screen for 5 seconds while the LLM generates.

```python
def chat(message, history):
    response = ""
    for chunk in llm.stream(message):
        response += chunk
        yield response  # Gradio renders incrementally
```

### Production Gradio Concerns

| Concern | What to do |
|---------|-----------|
| **Auth** | Gradio has built-in `auth=("user", "pass")` for prototypes. Production: put it behind nginx + OAuth. |
| **Concurrency** | `demo.launch(max_threads=N)` — each user request takes a thread. |
| **State per user** | Use `gr.State()` for per-session state. Don't use module-level globals. |
| **File uploads** | `gr.File()` — be careful with size limits and validation. |
| **Sharing** | `share=True` gives a public URL via Gradio's relay. **Never use in production** — leaks data, no auth. |
| **Deployment** | Gradio apps can run on HuggingFace Spaces, or as a Python process behind any reverse proxy. |

### Gradio vs Alternatives

| Tool | Best for |
|------|---------|
| **Gradio** | ML demos, internal tools, fast prototypes |
| **Streamlit** | Data apps, dashboards |
| **FastAPI + React** | Real production product |
| **Chainlit** | Chat-first apps with streaming, message history |

---

## 11. LangGraph vs Custom Orchestration

### What LangGraph Provides

LangGraph is a state-machine framework specifically designed for multi-agent workflows.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("profile", node_generate_profile)
graph.add_node("retrieve", node_retrieve_candidates)
graph.add_node("trends", node_analyze_trends)
# ...

graph.set_entry_point("profile")
graph.add_edge("profile", "retrieve")
graph.add_edge("retrieve", "trends")  # implicitly parallel if you add multiple
graph.add_edge("trends", "synthesis")
graph.add_edge("synthesis", END)

app = graph.compile()
result = app.invoke({"user_input": "..."})
```

### What You Get For Free

| Feature | Custom code | LangGraph |
|---------|------------|-----------|
| State management | Manual dict | Automatic merging |
| Parallel execution | ThreadPoolExecutor | Built-in |
| Checkpointing | Manual save/load | Built-in (SQLite, Postgres) |
| Human-in-the-loop | DIY | `interrupt_before` flag |
| Conditional edges | If-statements | `add_conditional_edges` |
| Visualization | None | `graph.get_graph().draw_png()` |
| Streaming | DIY | `app.stream()` yields per-node |

### When to Use Custom (Like Our Code)

- Simple workflows (3-6 nodes)
- You don't need persistence yet
- You want full control / no dependency
- Learning purposes

### When to Use LangGraph

- 10+ nodes with conditional branching
- Need checkpointing (resume after crash)
- Need human approval gates
- Need to swap models per node easily
- Production deployment with monitoring

**Our Module 3 implementation chose custom code** for clarity. In production, switching to LangGraph is a 2-hour refactor that buys you checkpointing, observability, and easier branching.

---

## 12. Production Concerns

### 12.1 Error Handling & Cascading Failures

In a 6-agent pipeline, a single agent failure can cascade:
- Profile agent fails → retrieval has no profile → synthesis has nothing to recommend.

**Pattern: Each agent has a fallback default**

```python
def node_analyze_trends(state):
    try:
        result = call_agent("food_trend_analyst", build_prompt(state))
        return {"trend_analysis": json.loads(result)}
    except (json.JSONDecodeError, APIError) as e:
        log.error(f"trend_analyst failed: {e}")
        return {"trend_analysis": {"trends": [], "error": str(e)}}
        # Pipeline continues — synthesis just won't have trend data
```

**Pattern: Circuit breaker**

If an agent fails N times in a row, stop calling it for M minutes:

```python
from circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@breaker
def call_trend_agent(state):
    return call_agent("food_trend_analyst", state)
```

### 12.2 Cost Management

Every agent = LLM call = $$$. With 100K daily users × 6 agents × $0.01/call = $6,000/day.

**Strategies:**

1. **Model routing** — Haiku for simple agents, Sonnet for synthesis
   ```python
   model_map = {
       "user_profile_generator": "haiku",
       "rag_retriever": "haiku",
       "recommendation_expert": "sonnet",  # complex synthesis
   }
   ```

2. **Prompt caching** — cache long system prompts
   ```python
   client.messages.create(
       system=[{
           "type": "text",
           "text": LONG_BACKSTORY,
           "cache_control": {"type": "ephemeral"}  # 5-min cache, 90% cheaper hits
       }]
   )
   ```

3. **Result caching** — same user query → return cached result
   ```python
   cache_key = hash(user_input + str(preferences))
   if cache_key in redis:
       return redis.get(cache_key)
   ```

4. **Lazy execution** — skip agents whose output won't be used
   - User asks "show me Italian restaurants" → skip Nutrition agent

### 12.3 Latency Optimization

**Sources of latency in our system:**
1. Time to first token (~200ms per call)
2. Output tokens × ~30ms/token
3. Network round-trip

**Optimizations:**

| Optimization | Latency reduction |
|-------------|-------------------|
| Parallelize Phase 3 | 3× → 1× wall-clock |
| Stream final output | Perceived latency drops to ~500ms |
| Use Haiku where possible | ~3× faster than Sonnet |
| Cap `max_tokens` | Forces shorter outputs |
| Prompt caching | ~85% latency drop on cached portion |
| Edge deployment (region match) | Cuts ~100ms |

### 12.4 Observability & Tracing

You can't fix what you can't see. Production multi-agent systems must log:

```python
import logging, time

def call_agent(agent_key, message):
    start = time.time()
    response = _get_client().messages.create(...)
    duration = time.time() - start

    logger.info({
        "agent": agent_key,
        "duration_ms": int(duration * 1000),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": response.model,
        "trace_id": current_trace_id(),
    })
    return response
```

**Tools:**
- **LangSmith** (LangChain's tracer) — visualize multi-agent traces
- **Helicone** — drop-in proxy for any LLM API, gives you a dashboard
- **OpenTelemetry** — vendor-neutral tracing

### 12.5 Streaming Responses

For chat UX, streaming is mandatory. Users perceive 200ms first-token as "fast" even if total response takes 5s.

```python
def chat(message, history):
    response = ""
    with client.messages.stream(...) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                response += event.delta.text
                yield response  # Gradio re-renders
```

For multi-agent: stream the **final synthesis** agent only. Earlier agents produce JSON, not user-facing prose.

### 12.6 State Persistence & Resumability

If your 30-second pipeline crashes at second 25, you don't want to start over.

```python
# Save state after each phase
def checkpoint(state, phase):
    redis.set(f"workflow:{state['user_id']}:state", json.dumps(state))
    redis.set(f"workflow:{state['user_id']}:phase", phase)

# Resume on retry
def resume(user_id):
    state = json.loads(redis.get(f"workflow:{user_id}:state"))
    phase = redis.get(f"workflow:{user_id}:phase")

    if phase == "profile_generated":
        state = node_retrieve_candidates(state)  # skip profile, start at retrieval
    # ...
```

LangGraph does this automatically with `checkpointer=SqliteSaver(...)`.

### 12.7 Caching at the Agent Level

Different cache strategies for different agents:

| Agent | Cache strategy | TTL |
|-------|---------------|-----|
| User Profile Generator | Cache by user_id (deterministic input) | 24h |
| RAG Retriever | Cache by query embedding (high reuse) | 1h |
| Trend Analyst | Cache globally (everyone gets same trends) | 24h |
| Recommendation Synthesizer | **No cache** (combines unique state) | - |

```python
@lru_cache(maxsize=10000)
def get_trends_cached(date_str: str):  # cache key = date
    return call_agent("food_trend_analyst", ...)
```

### 12.8 Fallbacks & Graceful Degradation

What if Anthropic's API is down?

```python
def call_agent(agent_key, message):
    try:
        return call_anthropic(agent_key, message)
    except APIConnectionError:
        return call_openai_fallback(agent_key, message)  # use a different provider
    except Exception as e:
        return STATIC_FALLBACK_RESPONSES[agent_key]  # last resort: pre-canned reply
```

For our chatbot: if everything fails, return "I'm having trouble right now, please try again." — never crash.

### 12.9 Tool Use vs Pure Prompt Agents

We built **prompt-only agents** in Module 3 (no external tools). In production, agents typically have tools:

```python
tools = [
    {
        "name": "search_database",
        "description": "Search the restaurant database",
        "input_schema": {...}
    },
    {
        "name": "get_user_history",
        "description": "Fetch user's past orders",
        "input_schema": {...}
    }
]

response = client.messages.create(
    model="claude-sonnet-4-6@20260401",
    tools=tools,
    messages=[{"role": "user", "content": "Find me Italian restaurants I haven't tried"}]
)
# Response includes tool_use blocks — execute them, return results, loop until done
```

**When to add a tool:** when the agent needs information it can't infer (real-time data, user-specific data, external APIs).

---

## 13. Testing Multi-Agent Systems

### The Hard Part

Unit testing agents is hard because:
1. Outputs are stochastic
2. Each test costs $$$ (real LLM calls)
3. Asserting exact strings is brittle

### Testing Strategy: Layers

**Layer 1: Mock the LLM (fast, free)**
```python
from unittest.mock import patch

def test_intent_classification():
    with patch("module.llm.invoke") as mock_llm:
        mock_llm.return_value.content = "restaurant"
        assert classify_intent("where to eat?") == "restaurant"
```
Tests the **plumbing** (parsing, validation, routing) — not the LLM itself.

**Layer 2: Golden dataset evals (slow, cheap)**
```python
# tests/evals/intent_classification.jsonl
{"input": "where to eat tonight?", "expected": "restaurant"}
{"input": "how do I cook pasta?", "expected": "recipe"}
# ...

def test_eval_accuracy():
    correct = 0
    for case in load_eval_set():
        if classify_intent(case["input"]) == case["expected"]:
            correct += 1
    assert correct / len(eval_set) > 0.95  # 95% accuracy threshold
```
Run nightly. Tracks regressions when you change prompts.

**Layer 3: LLM-as-judge for subjective outputs**
```python
def test_recommendation_quality():
    result = run_workflow(test_user_input)

    judge_prompt = f"""
    User profile: {test_user_input}
    Recommendations: {result['final_recommendations']}

    Score 1-10: do recommendations match the user's preferences?
    """
    score = int(judge_llm.invoke(judge_prompt))
    assert score >= 7
```
Use a stronger model as a judge. Imperfect but scalable.

**Layer 4: Production traffic replay**
Log real production requests, replay them after a prompt change, diff outputs.

### Testing Parallel Code

```python
def test_phase_3_parallelism():
    start = time.time()
    state = run_workflow(test_input)
    duration = time.time() - start

    # If sequential, would be ~9s. Parallel should be ~3s.
    assert duration < 5, "Phase 3 should run in parallel"
```

### Testing Failure Modes

```python
def test_agent_failure_does_not_crash_pipeline():
    with patch("module.call_agent") as mock:
        mock.side_effect = APIError("simulated failure")
        # Pipeline should complete with error in trend_analysis
        result = run_workflow(test_input)
        assert "error" in result["trend_analysis"]
        assert result["final_recommendations"]  # synthesis still ran
```

---

## 14. Common Pitfalls

### Pitfall 1: Sharing Mutable State Across Threads

❌
```python
state = {}
def worker(x):
    state[x] = compute(x)  # race condition!
```

✅
```python
def worker(x, state_copy):
    new_state = dict(state_copy)
    new_state[x] = compute(x)
    return new_state
```

### Pitfall 2: Catching `Exception` Too Broadly

```python
try:
    ...
except Exception as e:  # catches EVERYTHING including KeyboardInterrupt
    pass
```
Better:
```python
try:
    ...
except (json.JSONDecodeError, APIError) as e:  # specific
    log.error(f"...: {e}")
```

### Pitfall 3: Trusting LLM JSON Output

❌ `data = json.loads(response)` — crashes when LLM adds markdown fences
✅ Strip fences first, validate with Pydantic, default on failure.

### Pitfall 4: Long-Running Synchronous LLM Calls in Web Handlers

```python
@app.route("/chat")
def chat():
    return run_workflow(...)  # 30s — blocks worker, times out
```
Fix: stream the response, or queue it (Celery) and return a job ID.

### Pitfall 5: No Token Budget

Without `max_tokens=...`, an agent can produce a 100K-token response and bankrupt your account on a single bad request.

### Pitfall 6: Stale System Prompts

You update an agent's prompt, but tests still pass because they hit a cached LLM response. Always invalidate caches when prompts change (include prompt hash in cache key).

### Pitfall 7: No Rate Limiting per User

A malicious or buggy user sends 1000 requests/sec, runs your bill to $10K. Add per-user rate limits at the edge (nginx, Cloudflare, or your auth layer).

### Pitfall 8: Ignoring Token Counts in State

```python
state["all_data"] = json.dumps(huge_object)  # 50K tokens
node_synthesis(state)  # OOM or context limit error
```
Trim state aggressively between phases. Pass minimal context.

### Pitfall 9: Hardcoding Sequential Logic When Parallel is Safe

A common mistake: writing `node_a(state); node_b(state); node_c(state)` for independent agents. Profile your pipeline — anywhere agents don't depend on each other, parallelize.

### Pitfall 10: Treating the Chatbot as the Only UI

Chat is great for ambiguous requests. Forms are great for structured data. We mix both via tabs. Don't force "add a restaurant" through conversation.

---

## 15. Real-World Scenarios

### Scenario 1: Latency Spike at 9am Every Day

**Symptom:** chatbot p99 latency goes from 2s to 30s every weekday at 9am.

**Diagnosis:** rate limit. Everyone arrives at the office, opens the app — concurrent calls exceed Anthropic's per-minute limit.

**Fix:**
1. Add concurrency limiter (semaphore at 50 req/sec)
2. Queue overflow with 10s timeout, return "high traffic, try again"
3. Long-term: switch to provisioned throughput tier

### Scenario 2: One Agent Fails Often

**Symptom:** Nutrition agent fails 5% of the time with malformed JSON.

**Diagnosis:** the agent's output schema is too complex. Long, nested, hard for LLM to produce reliably.

**Fix:**
1. Simplify schema (flatten nesting)
2. Switch from prompt-based JSON to tool use (forces valid output)
3. Add Pydantic retry loop (give the LLM 2 chances to fix)
4. If all else fails, default to empty nutrition analysis (don't crash)

### Scenario 3: Cost Doubled Last Month

**Symptom:** OpenAI bill went from $5K to $10K with same DAU.

**Diagnosis:** somebody changed the synthesis agent's prompt to include retrieved candidates twice (oversight in PR review). Token usage doubled.

**Fix:**
1. Add token usage alerts (PagerDuty if >2× baseline)
2. Add prompt diff in CI (PR comment showing token impact)
3. Cap `max_tokens` per agent

### Scenario 4: User Complains "It Forgot What I Said"

**Symptom:** user mentioned they're vegetarian in turn 1, gets meat recommendations in turn 5.

**Diagnosis:** chatbot doesn't pass `history` into intent classifier or preference extractor. Each turn is treated independently.

**Fix:**
```python
def classify_intent(message, history):
    full_context = format_history(history) + f"\nLatest: {message}"
    return llm.invoke(full_context)
```
Or maintain a persistent profile in `gr.State` that accumulates across turns.

### Scenario 5: Synthesis Agent Hallucinates Restaurant Names

**Symptom:** users report restaurants in recommendations don't exist.

**Diagnosis:** synthesis agent is generating creative names instead of using retrieved candidates.

**Fix:**
1. Tighter prompt: "ONLY recommend from the list below. Never invent."
2. Tool use with restaurant_id enum (force selection from candidates)
3. Post-validation: filter recommendations against actual DB; reject hallucinated names.

### Scenario 6: Onboarding a New Agent

You add a 7th agent: "Allergy Risk Assessor."

**Steps:**
1. Define role/goal/backstory
2. Define its input (subset of state) and output (new state field)
3. Decide: parallel with others (in Phase 3) or new phase?
4. Update synthesis agent's prompt to use new field
5. Update tests / eval suite
6. Roll out behind a feature flag (5% of traffic) → measure quality → ramp up

### Scenario 7: Replacing OpenAI with Claude (or vice versa)

**Done well, this is a 1-day refactor:**
1. Abstract the LLM call: `llm_invoke(system, message) -> str`
2. Swap the implementation
3. Re-run eval set, compare scores
4. A/B test in production

**Done badly,** you scatter `OpenAI()` calls across 50 files and it takes 2 weeks. Always abstract your provider.

---

## 16. All Interview Questions & Answers

### Q1: "Why split this into multiple agents instead of one big prompt?"

**Answer:** Five reasons:
1. **Specialization** — focused prompts perform better than generic ones
2. **Modularity** — change/test one agent without breaking others
3. **Parallelism** — independent agents run concurrently (3× speedup in our Phase 3)
4. **Cost control** — route simple agents to Haiku, complex synthesis to Sonnet
5. **Failure isolation** — if one agent fails, the rest can still produce partial results

The downside is more LLM calls per request and more orchestration code, but for any task with >2 distinct sub-steps, multi-agent wins on quality and observability.

---

### Q2: "How would you handle one agent failing without crashing the whole pipeline?"

**Answer:**
1. Wrap each agent in a try/except that returns a structured error in the state
   ```python
   try:
       state["trend_analysis"] = call_agent(...)
   except Exception as e:
       state["trend_analysis"] = {"error": str(e), "trends": []}
   ```
2. Downstream agents (synthesis) handle missing/empty fields gracefully
3. Log the failure for monitoring
4. Add a circuit breaker if failures are sustained
5. For the user: still return partial results with a note "couldn't analyze nutrition this time"

---

### Q3: "Why threads instead of asyncio for parallel agent calls?"

**Answer:**
- LLM calls are I/O-bound — Python's GIL releases during network I/O, so threads are concurrent for the wait portion.
- `ThreadPoolExecutor` is dead simple — one import, no `async` infection across the codebase.
- For Module 3 with 3 parallel agents, threads are the right tool.

**When asyncio wins:** if the entire codebase is async (FastAPI, async DB drivers), or you need 100s of concurrent requests where thread overhead matters.

**When neither works:** CPU-bound work (use multiprocessing).

---

### Q4: "How would you scale this to 10K concurrent users?"

**Answer:**
1. **Stateless server** — push all state to Redis/Postgres
2. **Async orchestration** — switch from threads to asyncio for higher concurrency per worker
3. **Rate limiting** — semaphore at the LLM provider level (per-account quota), token bucket per user
4. **Caching** — Redis-backed cache for common queries; prompt caching at provider level
5. **Queue overflow** — Celery/SQS for async workflows; return job ID, poll for result
6. **Model routing** — Haiku for trivial agents, Sonnet only for synthesis
7. **CDN for static UI** — Gradio replaced with React + serverless API
8. **Horizontal scale** — auto-scale FastAPI workers behind load balancer
9. **Observability** — distributed tracing (OpenTelemetry), per-user latency histograms

---

### Q5: "What is shared state and why does each node return a new state?"

**Answer:** Shared state is a single dictionary that flows through every agent — input on the way in, accumulated outputs on the way out.

Why each node returns a (new) state:
- **Single source of truth** — at any point you can dump state and see exactly what's been computed
- **Serializable** — checkpoint to disk for resumability
- **Testable** — given input state, assert output state
- **Immutable-style** — copying state before mutation prevents race conditions in parallel execution

In LangGraph, this is enforced — nodes must return state updates.

---

### Q6: "How do you test agents that produce non-deterministic output?"

**Answer:** Three layers:
1. **Mock the LLM** for plumbing tests — does parsing, validation, routing work?
2. **Eval datasets** — golden input/expected pairs; assert >X% pass rate; nightly runs
3. **LLM-as-judge** — use a stronger model to score subjective outputs (relevance, helpfulness)

For exact-match tests on stochastic outputs, set `temperature=0` and pin model versions, but accept that small variations may still occur.

---

### Q7: "What's the difference between an agent and an LLM call?"

**Answer:**
- An **LLM call** is one prompt → one response. No memory, no tools, no decision-making about what to do next.
- An **agent** wraps an LLM with a defined role, goal, and (optionally) tools. The agent can decide which tools to invoke, how to combine their outputs, and when to stop.

In our Module 3 system, each "agent" is really a prompt-only agent (LLM + role/goal/backstory), but the abstraction lets us swap in tool-using agents later without changing the orchestration layer.

---

### Q8: "How would you implement the multi-agent system with LangGraph instead of custom code?"

**Answer:**
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("profile", node_generate_profile)
graph.add_node("retrieve", node_retrieve_candidates)
graph.add_node("trends", node_analyze_trends)
graph.add_node("styles", node_analyze_styles)
graph.add_node("nutrition", node_evaluate_nutrition)
graph.add_node("synthesize", node_generate_recommendations)

graph.set_entry_point("profile")
graph.add_edge("profile", "retrieve")
# Parallel: all three start after retrieve
graph.add_edge("retrieve", "trends")
graph.add_edge("retrieve", "styles")
graph.add_edge("retrieve", "nutrition")
# All three must finish before synthesize
graph.add_edge(["trends", "styles", "nutrition"], "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile(checkpointer=SqliteSaver.from_conn_string("checkpoints.db"))
result = app.invoke({"user_input": "..."}, config={"thread_id": user_id})
```

Benefits over custom: free checkpointing, visualization, conditional edges, streaming per node.

---

### Q9: "How does intent classification work and how would you make it more reliable?"

**Answer:**
Intent classification uses an LLM to map free-form text to one of N categories.

To make it more reliable:
1. **Tool use / function calling** — force the LLM to pick from an enum
2. **Few-shot examples** — include 3-5 examples in the prompt
3. **Validate output** — if model returns garbage, default to "clarification"
4. **Confidence threshold** — use logprobs to detect uncertain classifications and route to clarification
5. **Long-term: train a small classifier** on collected (text, intent) pairs for the high-volume path; LLM only for edge cases

---

### Q10: "Why have separate forms (Add Restaurant, Add Recipe) when you have a chat?"

**Answer:**
Chat is great for **ambiguous, exploratory** requests ("what should I eat tonight?"). Forms are better for **structured, high-precision** input (a restaurant has 5 specific fields, none optional).

Forcing structured data through chat creates two problems:
1. **User friction** — "Now tell me the price range, now the cuisine, now the location..." takes 6 turns vs filling 5 fields at once.
2. **LLM extraction errors** — every slot you parse from prose has ~5% error rate; that compounds.

The pattern: **let users discover via chat, structure data via forms.**

---

### Q11: "How would you debug a slow chatbot in production?"

**Answer:**
1. **Add tracing** — log timestamp + duration for every LLM call (OpenTelemetry, LangSmith)
2. **Find the slow span** — synthesis agent? Retrieval? Network?
3. **Check token counts** — output tokens × ~30ms each; max_tokens too high?
4. **Check parallelism** — Phase 3 actually parallel, or accidentally sequential?
5. **Check rate limits** — provider 429s causing retries with backoff?
6. **Check caching** — prompt cache hits or misses?
7. **Check streaming** — is the user perceiving the delay because we're not streaming?

Most production "slow" complaints are actually **time-to-first-token** issues — fix with streaming first.

---

### Q12: "How do you stop the LLM from generating outputs that don't match your JSON schema?"

**Answer:** Three approaches, in order of robustness:
1. **Tool use / structured output mode** — provider enforces the schema via constrained decoding. ~99% reliable. (Anthropic's `tools` parameter, OpenAI's `response_format={"type": "json_schema", ...}`).
2. **Pydantic + retry loop** — parse, on `ValidationError` feed the error back and ask the model to fix. ~98% reliable, slower.
3. **Plain JSON in prompt** — works ~95% of the time. Always have a fallback default.

For production, use #1. Fall back to #2 if your provider doesn't support it.

---

### Q13: "How would you add a 7th agent to the system?"

**Answer:**
1. **Define** role, goal, backstory
2. **Define** input (which state fields it reads) and output (which field it writes)
3. **Decide placement** — does it fit in an existing phase (Phase 3 parallel)? Or does it depend on someone's output? (sequential)
4. **Update downstream agents** that should consume its output (typically synthesis)
5. **Update tests** — golden dataset for the new agent; integration tests for the full pipeline
6. **Add monitoring** — log its latency, error rate, output size
7. **Roll out behind feature flag** — 5% of traffic, measure recommendation quality, ramp up

---

### Q14: "What's prompt caching and when should I use it?"

**Answer:** Prompt caching lets the LLM provider cache portions of your prompt server-side, so subsequent calls only re-process the changed part.

```python
client.messages.create(
    system=[
        {"type": "text", "text": LONG_BACKSTORY, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": short_dynamic_part}
    ]
)
```

**Use it when:**
- You have a long, stable system prompt (>1024 tokens for Anthropic)
- You make repeated calls within ~5 minutes (TTL window)
- Same agent invoked many times per session

**Savings:** ~90% reduction in input token cost on cache hits, ~85% latency reduction.

---

### Q15: "What happens if the LLM provider's API goes down?"

**Answer:**
1. **Retries with exponential backoff** — handle transient 503s automatically
2. **Multi-provider fallback** — if Anthropic is down, route to OpenAI/Bedrock
3. **Circuit breaker** — after N failures, stop hammering the dead endpoint for M minutes
4. **Static fallback** — last resort: return a canned "I'm having trouble, please try later" message
5. **Status page** — proactively show users a banner if a partial outage is happening

Multi-provider is hardest because prompts often need tuning per model. Worth it for high-availability products.

---

## 17. Concepts You Must Know

### Quick Reference Table

| Concept | One-line explanation |
|---------|---------------------|
| **AI Agent** | LLM + role + goal + (optional) tools — a goal-directed component |
| **Multi-Agent** | Multiple specialized agents collaborating on a complex task |
| **Sequential pipeline** | Agent A → Agent B → Agent C, each consumes prior output |
| **Parallel fan-out** | Independent agents run concurrently; results merged |
| **Hybrid workflow** | Mix of sequential phases + parallel agents within phases (our Module 3) |
| **Hierarchical** | Manager agent routes work to specialized workers |
| **ReAct loop** | Single agent loops: reason → act → observe |
| **Shared state** | Single dict flowing through all nodes; readable + writable |
| **TypedDict** | Lightweight typed dict for state schema (LangGraph default) |
| **Role / Goal / Backstory** | The 3 fields that define an agent's identity and steer its outputs |
| **Tool use** | Forcing LLM to call structured functions instead of free text |
| **ThreadPoolExecutor** | Python's thread pool — used for I/O-bound parallelism (LLM calls) |
| **Intent classification** | Mapping free-form input to one of N action categories |
| **Preference extraction** | Converting prose into structured JSON (NER-style) |
| **Cascading failure** | One agent's bad output corrupts downstream agents |
| **Circuit breaker** | Stop calling a failing service to prevent cascade |
| **Streaming** | Yielding tokens as they're generated → faster perceived response |
| **LangGraph** | State-machine framework for multi-agent workflows |
| **Prompt caching** | Server-side cache of long prompts → cheaper, faster repeat calls |
| **LLM-as-judge** | Use a stronger model to score outputs of weaker models (eval pattern) |
| **Tool use / function calling** | LLM produces structured calls to your functions; output schema enforced |

### The Module 3 System in One Diagram

```
                   Gradio UI (Module 3 Ex 3)
                   ──────────────────────────
        [Chat tab]          [Add Restaurant tab]    [Add Recipe tab]
            │                       │                       │
            ▼                       ▼                       ▼
       classify_intent          add_restaurant          add_recipe
            │
       ┌────┴────┬─────────┬──────────┬─────────┐
       ▼         ▼         ▼          ▼         ▼
  restaurant  recipe     both    clarification database
       │         │         │          │         │
       └────┬────┴─────────┘     help text   redirect
            ▼
     extract_preferences
            │
            ▼
     ┌─────────────────────────────────────────────┐
     │  Multi-Agent Workflow (Module 3 Ex 2)        │
     │                                              │
     │  Phase 1 (seq):  User Profile Generator      │
     │  Phase 2 (seq):  RAG Retriever               │
     │  Phase 3 (par):  ┌─ Trend Analyst    ─┐      │
     │                  ├─ Style Expert     ─┤      │
     │                  └─ Nutrition Expert ─┘      │
     │  Phase 4 (seq):  Recommendation Synthesizer  │
     └─────────────────────────────────────────────┘
            │
            ▼
     format_recommendations  →  display in chat
```

### The Three Latency Equations

```
1. Time per agent:
   t_agent = TTFT + (output_tokens × t_per_token)
           ≈ 200ms + (500 tokens × 30ms) = 15.2s   (Sonnet)
           ≈ 100ms + (500 tokens × 10ms) =  5.1s   (Haiku)

2. Sequential chain of N agents:
   t_total = sum(t_agent_i for i in 1..N)

3. Parallel fan-out of N agents:
   t_total = max(t_agent_i for i in 1..N)
```

### Decision Flowchart: Should I Use Multi-Agent?

```
Is the task >1 distinct subtask?
│
├─ NO  → Single LLM call. Don't over-engineer.
│
└─ YES → Are the subtasks independent?
         │
         ├─ NO  → Sequential pipeline.
         │
         └─ YES → Parallel fan-out (with synthesis if results need merging).
```

### The Production Readiness Checklist

When promoting a multi-agent system to production:

- [ ] Each agent has try/except with structured error fallback
- [ ] All LLM calls have `max_tokens` set
- [ ] All JSON outputs go through Pydantic or tool-use validation
- [ ] State has a `workflow_step` field for tracing
- [ ] Logging captures: agent_name, duration, input_tokens, output_tokens, trace_id
- [ ] Per-user rate limit at the edge
- [ ] Provider rate limit (semaphore) inside the app
- [ ] Streaming for the user-facing final response
- [ ] Eval suite runs in CI with accuracy threshold
- [ ] Token cost dashboard with budget alerts
- [ ] Fallback model / provider configured for outages
- [ ] Prompts version-controlled; changes require PR review
- [ ] System prompt cached if >1024 tokens
- [ ] No secrets in prompts (no API keys, PII in system messages)
