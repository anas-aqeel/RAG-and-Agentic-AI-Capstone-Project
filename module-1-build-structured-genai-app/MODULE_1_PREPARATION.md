# Module 1 — Interview Preparation & Deep Dive

> **What this module covered:** Using LLMs to transform unstructured data (text + images) into structured JSON, validating outputs, building a CLI application, and testing LLM-dependent code.

---

## Table of Contents

1. [Scaling LLM Pipelines (109 → 1M)](#1-scaling-llm-pipelines-from-109-to-1-million)
2. [Testing LLM-Dependent Code](#2-testing-code-that-calls-an-llm)
3. [Data Safety: Backup & Atomic Writes](#3-why-backup-before-writing--atomic-writes)
4. [Pattern Deep Dives](#4-pattern-deep-dives)
   - [Structured Output from LLMs](#41-structured-output-from-llms)
   - [Retry & Auto-Repair Strategy](#42-retry--auto-repair-strategy)
   - [Multimodal Pipelines](#43-multimodal-pipelines)
   - [Data Validation with Pydantic](#44-data-validation-with-pydantic)
   - [Cost Control & Model Routing](#45-cost-control--model-routing)
   - [Idempotency](#46-idempotency)
5. [All Interview Questions & Answers](#5-all-interview-questions--answers)
6. [Concepts You Must Know](#6-concepts-you-must-know)

---

## 1. Scaling LLM Pipelines (from 109 to 1 Million)

### The Question

> "How would you scale this from 109 recipes to 1M?"

### Short Answer

Use **async/concurrent API calls**, **batching**, **caching identical prompts**, **checkpointing progress**, and **model routing** (cheap model first, expensive only on failure).

### Deep Explanation

#### The Problem

In our Exercise 1, we processed 109 restaurants **sequentially** — one API call at a time. Each call takes ~1-3 seconds. At 1M records:

```
1,000,000 records × 2 seconds/call = 2,000,000 seconds = ~23 DAYS
```

That's not viable. Here's how industry solves it:

---

#### Strategy 1: Async / Concurrent Processing

**Analogy:** Imagine you're at a restaurant. Sequential processing = ordering one dish, waiting for it to arrive, eating it, then ordering the next. Async = ordering 50 dishes at once and eating them as they arrive.

**How it works:**

```python
import asyncio
from anthropic import AsyncAnthropicVertex

client = AsyncAnthropicVertex(project_id="my-project", region="us-east5")

async def process_one(restaurant_text, semaphore):
    async with semaphore:  # Limit concurrent requests
        response = await client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": restaurant_text}]
        )
        return response.content[0].text

async def process_all(restaurants):
    semaphore = asyncio.Semaphore(50)  # Max 50 concurrent calls
    tasks = [process_one(r, semaphore) for r in restaurants]
    return await asyncio.gather(*tasks)
```

**Result:** With 50 concurrent calls, 1M records at 2s each = ~11 hours (vs 23 days).

**Industry standard:**
- Use `asyncio.Semaphore` to respect API rate limits
- Most LLM APIs allow 50-200 concurrent requests
- Cloud platforms (Vertex AI, AWS Bedrock) have per-minute token quotas

---

#### Strategy 2: Batching

**Analogy:** Instead of mailing 1M individual letters, you pack them into boxes of 100 and ship the boxes.

**How it works:**

Some APIs support batch endpoints where you submit thousands of requests at once and get results later (hours, not real-time).

```python
# Anthropic's Message Batches API
from anthropic import Anthropic

client = Anthropic()

batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"restaurant-{i}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": text}]
            }
        }
        for i, text in enumerate(restaurant_texts)
    ]
)
# Results arrive later — poll or use webhook
```

**Benefits:**
- 50% cheaper on Anthropic's batch API
- No rate limit pressure
- Results within 24 hours

**When to use:** When you don't need real-time results. Perfect for bulk data processing like our restaurant structuring task.

---

#### Strategy 3: Caching

**Analogy:** If 500 students ask the same math question, the teacher writes the answer on the board once instead of answering 500 times.

**Types of caching in LLM pipelines:**

```python
import hashlib
import json
import redis

cache = redis.Redis()

def llm_with_cache(system_msg, prompt):
    # Create a deterministic cache key
    cache_key = hashlib.sha256(
        f"{system_msg}||{prompt}".encode()
    ).hexdigest()

    # Check cache first
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Cache miss — call the LLM
    result = call_llm(system_msg, prompt)
    cache.set(cache_key, json.dumps(result), ex=86400)  # 24h TTL
    return result
```

**Real-world scenario:** If 50 restaurants have nearly identical descriptions, or you're reprocessing after a crash, caching prevents redundant API calls and saves money.

**Anthropic Prompt Caching:** Anthropic also supports server-side prompt caching where the system prompt/context is cached across calls:

```python
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    system=[{
        "type": "text",
        "text": long_system_prompt,
        "cache_control": {"type": "ephemeral"}  # Cache this part
    }],
    messages=[{"role": "user", "content": unique_part}]
)
# The system prompt is cached for 5 min — subsequent calls are 90% cheaper on input tokens
```

---

#### Strategy 4: Checkpointing

**Analogy:** Saving your game progress. If you die at level 50, you don't restart from level 1.

```python
import json

CHECKPOINT_FILE = "progress.json"

def load_checkpoint():
    try:
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"processed": [], "last_index": 0}

def save_checkpoint(results, index):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": results, "last_index": index}, f)

# Resume from where we left off
checkpoint = load_checkpoint()
for i in range(checkpoint["last_index"], len(restaurants)):
    result = process(restaurants[i])
    checkpoint["processed"].append(result)
    if i % 100 == 0:  # Save every 100 records
        save_checkpoint(checkpoint["processed"], i + 1)
```

**Why it matters:** At 1M records, if your script crashes at record 800K, you don't want to redo 800K API calls ($$$).

---

#### Strategy 5: Model Routing (Cascade)

**Analogy:** In a hospital, a nurse handles simple cases. Only complex ones go to the specialist (expensive).

```python
def process_with_cascade(text):
    # Try cheap model first
    result = call_llm(model="haiku", text=text)
    try:
        Restaurant.model_validate_json(result)
        return result  # Haiku succeeded — saved money
    except ValidationError:
        # Fall back to expensive model
        result = call_llm(model="sonnet", text=text)
        return result
```

**Cost impact:**
- Haiku: $0.80/1M input tokens
- Sonnet: $3.00/1M input tokens
- If Haiku handles 90% of cases, you save ~70% on costs

---

#### Complete Scaling Architecture

```
                    ┌─────────────────┐
                    │  1M Raw Records  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Check Cache     │──── Hit ──→ Skip (free)
                    └────────┬────────┘
                             │ Miss
                    ┌────────▼────────┐
                    │  Haiku (cheap)   │──── Valid ──→ Done
                    └────────┬────────┘
                             │ Invalid
                    ┌────────▼────────┐
                    │  Sonnet (smart)  │──── Valid ──→ Done
                    └────────┬────────┘
                             │ Still Invalid
                    ┌────────▼────────┐
                    │  Error Queue     │──→ Manual review
                    └─────────────────┘

    All of the above running with 50 concurrent workers
    Checkpointing every 100 records
```

---

### Related Issues at Scale

| Issue | Solution |
|-------|----------|
| API rate limits (429 errors) | Exponential backoff + semaphore |
| Token quota exceeded | Spread across multiple API keys/regions |
| Memory (1M records in RAM) | Stream processing, process in chunks of 10K |
| Inconsistent outputs | Temperature=0, seed parameter for reproducibility |
| Cost monitoring | Track tokens per request, set budget alerts |
| Network failures | Retry with backoff, checkpoint progress |

---

## 2. Testing Code That Calls an LLM

### The Question

> "How do you test code that calls an LLM?"

### Short Answer

Use **mocking** to isolate LLM calls, **dependency injection** to swap implementations, **snapshot testing** to detect output drift, and **eval frameworks** to measure quality at scale.

### Deep Explanation

#### The Problem

LLM calls are:
- **Non-deterministic:** Same input can give different outputs
- **Slow:** 1-5 seconds per call
- **Expensive:** Every test run costs money
- **External:** Depend on a third-party API being available

You can't have your CI/CD pipeline making real LLM calls on every git push.

---

#### Technique 1: Mocking (What We Did)

**Analogy:** A flight simulator. Pilots don't learn by crashing real planes — they use a mock that behaves like a plane.

```python
from unittest.mock import patch, MagicMock

def test_new_data_entry():
    fake_llm_response = '{"name": "Test Cafe", "location": "NYC", ...}'

    with patch('restaurant_data_management.llm_model') as mock_llm:
        mock_llm.return_value = fake_llm_response
        result = new_data_entry_process("Some paragraph", 1000001)

    assert result["name"] == "Test Cafe"
    assert result["itemId"] == 1000001
    mock_llm.assert_called()  # Verify LLM was called
```

**What this tests:** Your parsing logic, validation, retry loop — everything EXCEPT the LLM itself.

**Limitation:** You're assuming the LLM returns what you expect. If the real LLM starts returning a different format, your mock won't catch it.

---

#### Technique 2: Dependency Injection

**Analogy:** A car engine that accepts any fuel type. You inject diesel for production, but test fuel in the lab.

```python
# Instead of hardcoding the LLM call inside the function:
class RestaurantProcessor:
    def __init__(self, llm_fn):
        self.llm = llm_fn  # Inject the LLM function

    def process(self, paragraph, item_id):
        response = self.llm(system_msg, paragraph)
        # ... validation logic ...

# Production:
processor = RestaurantProcessor(llm_fn=real_claude_call)

# Testing:
def fake_llm(system_msg, prompt):
    return '{"name": "Mock Restaurant", ...}'

processor = RestaurantProcessor(llm_fn=fake_llm)
result = processor.process("test paragraph", 1)
```

**Industry standard:** This is the preferred pattern in production systems. It makes code testable, swappable (change LLM providers easily), and clean.

---

#### Technique 3: Snapshot Testing

**Analogy:** Taking a photo of a building every month. If something changes, you notice immediately by comparing photos.

```python
import json

def test_llm_output_snapshot():
    # Run once with real LLM, save the output as a "golden" snapshot
    # On subsequent runs, compare against the snapshot

    prompt = "Extract restaurant data from: ..."
    result = llm_model(system_msg, prompt)

    # First run: save snapshot
    # snapshot = json.loads(result)
    # with open("snapshots/restaurant_test_1.json", "w") as f:
    #     json.dump(snapshot, f)

    # Subsequent runs: compare
    with open("snapshots/restaurant_test_1.json") as f:
        expected = json.load(f)

    actual = json.loads(result)

    # Don't compare exact strings — compare structure
    assert set(actual.keys()) == set(expected.keys())
    assert type(actual["rating"]) == type(expected["rating"])
```

**When to use:** When you want to detect if an LLM model update changes your outputs.

---

#### Technique 4: Eval Frameworks (Industry Standard)

**Analogy:** A standardized exam for your LLM. 100 questions with known-good answers, scored automatically.

```python
# Using a golden dataset
test_cases = [
    {
        "input": "The Gilded Artichoke is an upscale...",
        "expected_name": "The Gilded Artichoke",
        "expected_location": "Downtown",
        "expected_type": "upscale"
    },
    # ... 99 more cases
]

def evaluate():
    scores = {"name_match": 0, "schema_valid": 0, "total": len(test_cases)}

    for case in test_cases:
        result = process_restaurant(case["input"])
        if result and result.get("name") == case["expected_name"]:
            scores["name_match"] += 1
        if result:
            scores["schema_valid"] += 1

    print(f"Name accuracy: {scores['name_match']}/{scores['total']}")
    print(f"Schema validity: {scores['schema_valid']}/{scores['total']}")
```

**Industry tools:**
- **DeepEval** — LLM testing framework with metrics like faithfulness, relevance
- **RAGAS** — Evaluation for RAG pipelines (we'll use this in Module 2)
- **LangSmith** — Tracing + evaluation by LangChain
- **Promptfoo** — Open-source prompt testing

---

#### Testing Pyramid for LLM Apps

```
        ▲
       /  \        Eval Tests (weekly)
      / E  \       Real LLM, golden dataset, quality metrics
     /──────\
    /        \     Integration Tests (per PR)
   /   I      \    Real LLM, small sample, schema validation
  /────────────\
 /              \  Unit Tests (every commit)
/       U        \ Mocked LLM, test parsing/validation/retry logic
/──────────────────\
```

| Level | Real LLM? | Speed | Cost | What it catches |
|-------|-----------|-------|------|-----------------|
| Unit | No (mocked) | Fast (ms) | Free | Logic bugs, schema errors |
| Integration | Yes | Slow (s) | $ | API changes, prompt regressions |
| Eval | Yes | Very slow | $$$ | Quality drift, model updates |

---

## 3. Why Backup Before Writing & Atomic Writes

### The Question

> "Why backup before writing? What if the process crashes mid-write?"

### Short Answer

A crash during file write can **corrupt your entire database**. Backups + atomic writes ensure you never lose data.

### Deep Explanation

#### The Problem

```python
# DANGEROUS: What we did in exercise 3
with open("data.json", "w") as f:  # This EMPTIES the file immediately
    json.dump(data, f)             # If crash happens here → empty file, data lost
```

The moment you open a file with `"w"` mode, the OS **truncates it to zero bytes**. If your program crashes, loses power, or runs out of disk space before `json.dump` finishes, you have an empty or partially-written file.

**Real-life analogy:** Imagine you're rewriting a contract. You shred the original first, then start typing the new version. If the power goes out mid-typing, you've lost both versions.

---

#### What We Did (Basic Backup)

```python
def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)  # Step 1: copy the original
    with open(file_path, 'w') as f:           # Step 2: overwrite
        json.dump(data, f, indent=4)
```

**Problem:** If crash happens between Step 1 and Step 2, you still have the backup. But if crash happens DURING Step 2, you have a corrupted main file and a backup that's one version behind.

---

#### Industry Standard: Atomic Writes

**Analogy:** Instead of erasing the whiteboard and rewriting, you write the new version on a NEW whiteboard, then swap the whiteboards in one instant.

```python
import os
import tempfile
import json

def atomic_save(data, file_path):
    # Step 1: Write to a TEMPORARY file in the same directory
    dir_name = os.path.dirname(file_path) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(data, tmp, indent=4)
        tmp.flush()
        os.fsync(tmp.fileno())  # Force write to disk
        tmp_path = tmp.name

    # Step 2: Atomically replace the old file with the new one
    os.replace(tmp_path, file_path)  # This is atomic on most OS
```

**Why this works:**
- `os.replace()` is an **atomic operation** — it either fully succeeds or doesn't happen at all
- The old file exists right until the moment the new file takes its place
- If crash happens during write → temp file is garbage, original is untouched
- If crash happens during replace → OS guarantees atomicity

---

#### What Databases Do (WAL — Write-Ahead Log)

**Analogy:** Before a surgeon operates, they write down every step they'll perform. If something goes wrong, another surgeon can read the log and either finish or undo the operation.

```
1. Write intention to log: "I want to change record #42"
2. Write new data to log
3. Apply change to actual database
4. Mark log entry as "completed"

If crash after step 2: Read log, redo step 3
If crash after step 3: Read log, mark as completed
```

SQLite, PostgreSQL, MySQL — they ALL use WAL. It's the gold standard for data integrity.

---

#### Comparison of Strategies

| Strategy | Data Loss Risk | Complexity | When to Use |
|----------|---------------|------------|-------------|
| No backup (just overwrite) | HIGH — one crash = total loss | None | Never in production |
| Copy backup before write | MEDIUM — backup is 1 version behind | Low | Simple scripts, our exercise |
| Atomic write (temp + replace) | LOW — only lose in-flight write | Medium | Single-file databases, config |
| Write-Ahead Log (WAL) | NEAR ZERO | High | Real databases |
| Database (SQLite/Postgres) | NEAR ZERO | Medium | Any serious application |

---

#### Related Issues

**Issue: Concurrent writes**
If two users run the CLI at the same time, both read the file, both write — one overwrites the other.
**Solution:** File locking (`fcntl.flock`) or use a proper database.

**Issue: Disk full**
`json.dump` fails midway because disk is full.
**Solution:** Atomic write pattern — temp file fails, original stays intact.

**Issue: Large files**
A 1GB JSON file takes seconds to write. Long write = bigger crash window.
**Solution:** Use a database (SQLite). Or split into multiple smaller files.

---

## 4. Pattern Deep Dives

### 4.1 Structured Output from LLMs

#### What We Did
```python
# Prompt the LLM and hope it returns valid JSON
system_msg = "Output ONLY the JSON object, nothing else."
response = llm_model(system_msg, prompt)
data = json.loads(response)  # Might fail if LLM adds commentary
```

#### The Problem
LLMs are probabilistic text generators. Even with "output ONLY JSON", they sometimes:
- Add ```json markdown wrappers
- Include explanatory text before/after the JSON
- Use single quotes instead of double quotes
- Hallucinate fields not in your schema
- Omit required fields

#### Industry Solutions

**Solution 1: Tool Use / Function Calling**

Instead of asking the LLM to "output JSON", you define a tool with a JSON schema. The LLM is forced to call the tool, which guarantees valid JSON.

```python
import anthropic

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    tools=[{
        "name": "save_restaurant",
        "description": "Save structured restaurant data",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {"type": "string"},
                "rating": {"type": "number"},
                "cuisine": {"type": "string"}
            },
            "required": ["name", "location", "cuisine"]
        }
    }],
    messages=[{"role": "user", "content": f"Extract: {paragraph}"}]
)

# response.content[0].input is GUARANTEED valid JSON matching the schema
restaurant = response.content[0].input
```

**Why this is better:** The API layer enforces the schema before returning. No parsing errors, no markdown wrappers, no hallucinated fields.

**Solution 2: Constrained Decoding (e.g., Outlines, Guidance)**

Force the LLM to only generate tokens that form valid JSON:

```python
# Using the 'outlines' library
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B")
generator = outlines.generate.json(model, Restaurant)  # Pydantic model
result = generator("Extract restaurant data from: ...")
# result is ALWAYS a valid Restaurant object
```

**Analogy:** Instead of asking someone to write a number between 1-10 and hoping they comply, you give them a dial that only goes from 1 to 10.

---

### 4.2 Retry & Auto-Repair Strategy

#### What We Did

```python
for retries in range(4):
    try:
        Restaurant.model_validate_json(candidate)
        break
    except ValidationError as e:
        repair_prompt = f"Fix this JSON: {candidate}\nError: {e}"
        candidate = llm_model(repair_system, repair_prompt)
```

**Analogy:** Submitting a tax form. IRS rejects it with specific errors. You fix those exact errors and resubmit. Repeat until accepted.

#### Industry Patterns

**Exponential Backoff:**
```python
import time

for attempt in range(5):
    try:
        result = call_api()
        break
    except RateLimitError:
        wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
        time.sleep(wait)
```

**Circuit Breaker:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED = normal, OPEN = blocked

    def call(self, fn, *args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "HALF_OPEN"  # Try one request
            else:
                raise Exception("Circuit is OPEN — API is down")

        try:
            result = fn(*args)
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise
```

**Analogy:** A fuse in your house. If too many appliances blow the fuse (failures), the circuit opens (stops all requests). After a cooldown, you try again. This prevents hammering a dead API.

**When to use what:**

| Pattern | Use When |
|---------|----------|
| Simple retry | Transient errors (network blip) |
| Exponential backoff | Rate limiting (429 errors) |
| Auto-repair (our approach) | LLM output validation failures |
| Circuit breaker | Repeated failures, API outages |
| Dead letter queue | After max retries, save for manual review |

---

### 4.3 Multimodal Pipelines

#### What We Did

```python
# Sequential: one image at a time, base64 encoded
for recipe in recipes:
    image_bytes = open(f"recipe{recipe['id']}.png", "rb").read()
    image_b64 = base64.b64encode(image_bytes).decode()
    caption = vision_llm(system_msg, prompt, image_b64)
    recipe["image_description"] = caption
```

#### Industry Improvements

**Image Preprocessing (save tokens & cost):**

```python
from PIL import Image

def preprocess_image(path, max_size=1024):
    """Resize large images before sending to LLM.
    A 4000x3000 image costs more tokens than a 1024x768 one,
    with minimal quality difference for captioning."""
    img = Image.open(path)
    img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
```

**Why:** LLM vision APIs charge per image token. A 4K image might use 2000 tokens. A resized 1024px image uses ~800 tokens — same caption quality, 60% cheaper.

**Async Multimodal Processing:**

```python
async def caption_all_images(recipes):
    semaphore = asyncio.Semaphore(20)  # Limit concurrent vision calls

    async def caption_one(recipe):
        async with semaphore:
            img = preprocess_image(f"recipe{recipe['id']}.png")
            return await async_vision_llm(prompt, img)

    tasks = [caption_one(r) for r in recipes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for recipe, result in zip(recipes, results):
        if isinstance(result, Exception):
            recipe["image_description"] = ""  # Graceful degradation
        else:
            recipe["image_description"] = result
```

**Analogy:** Instead of one photographer captioning one photo at a time, you hire 20 photographers who all work simultaneously, but you limit it to 20 because the office only has 20 desks.

---

### 4.4 Data Validation with Pydantic

#### What We Did

```python
class Restaurant(BaseModel):
    name: str
    location: str
    rating: Optional[float] = None
    signatures: List[str] = Field(default_factory=list)

# Validate LLM output
Restaurant.model_validate_json(llm_response)
```

#### Why Pydantic Over Plain `json.loads()`

```python
# json.loads only checks if it's valid JSON syntax
data = json.loads('{"name": 123}')  # Succeeds! But name should be a string

# Pydantic checks types, required fields, constraints
Restaurant.model_validate_json('{"name": 123}')
# ValidationError: name — Input should be a valid string
```

**Analogy:** `json.loads` checks if a document is printed on real paper. Pydantic checks if the document has all required sections, signatures, and correct dates.

#### Advanced Pydantic Features for LLM Outputs

```python
from pydantic import BaseModel, Field, field_validator

class Restaurant(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    rating: float = Field(ge=0, le=5)  # Between 0-5
    price_range: int = Field(ge=1, le=4)  # 1-4 dollar signs

    @field_validator("name")
    @classmethod
    def name_not_placeholder(cls, v):
        if v.lower() in ["n/a", "unknown", "test"]:
            raise ValueError("Name cannot be a placeholder")
        return v.strip()
```

#### Industry Alternatives

| Tool | Use Case |
|------|----------|
| **Pydantic** | Python-native, great for LLM output validation |
| **JSON Schema** | Language-agnostic, used in OpenAPI specs |
| **Great Expectations** | Data pipeline validation (check distributions, nulls) |
| **Zod** (TypeScript) | Same idea as Pydantic for JS/TS |
| **Guardrails AI** | Pydantic + auto-retry built specifically for LLM outputs |

---

### 4.5 Cost Control & Model Routing

#### What We Did

We used Haiku (cheap) for simple tasks. The CLAUDE.md specifies:

| Model | Cost (Input/Output per 1M tokens) | Use |
|-------|-----------------------------------|-----|
| Haiku 4.5 | $0.80 / $4.00 | Fast tasks, iteration |
| Sonnet 4.6 | $3.00 / $15.00 | Primary work |
| Opus 4.6 | $15.00 / $75.00 | Complex reasoning only |

#### Model Routing Pattern

```python
def smart_process(text, budget="low"):
    models = {
        "low": "claude-haiku-4-5@20251001",
        "medium": "claude-sonnet-4-6@20260401",
        "high": "claude-opus-4-6@20260401"
    }

    # Try cheap first
    for tier in ["low", "medium", "high"]:
        if tier == "high" and budget != "high":
            break  # Don't use Opus unless explicitly allowed

        result = call_llm(model=models[tier], text=text)
        try:
            validated = Restaurant.model_validate_json(result)
            return validated
        except ValidationError:
            continue  # Try next tier

    return None  # All tiers failed
```

**Analogy:** Customer support tiers. Level 1 (chatbot/Haiku) handles easy questions. Escalate to Level 2 (human/Sonnet) for complex ones. Level 3 (manager/Opus) only for critical issues.

#### Cost Calculation Example

Processing 1M restaurants:

| Strategy | Model Used | Cost |
|----------|-----------|------|
| All Opus | 1M × Opus | ~$90,000 |
| All Sonnet | 1M × Sonnet | ~$18,000 |
| All Haiku | 1M × Haiku | ~$4,800 |
| Routing (90% Haiku, 10% Sonnet) | Mixed | ~$6,100 |
| Routing + Caching (50% cache hit) | Mixed | ~$3,050 |

---

### 4.6 Idempotency

#### What It Means

**Idempotent** = Running the same operation multiple times produces the same result.

**Analogy:** An elevator button. Pressing "Floor 3" once or fifty times — you still go to floor 3. That's idempotent. A vending machine button is NOT idempotent — pressing it twice gives you two drinks.

#### Why It Matters

If your script crashes at record 500 and you restart, what happens to records 1-499?

**Non-idempotent (BAD):**
```python
data = []
for restaurant in restaurants:
    data.append(process(restaurant))  # Records 1-499 get duplicated on restart!
save(data)
```

**Idempotent (GOOD):**
```python
processed = load_checkpoint()  # Load what we already did
for i, restaurant in enumerate(restaurants):
    if i in processed:
        continue  # Skip already-processed records
    result = process(restaurant)
    processed[i] = result
    save_checkpoint(processed)
```

#### Industry Implementation

**Using unique IDs:**
```python
def process_restaurant(restaurant, output_db):
    item_id = restaurant["itemId"]

    # Check if already processed (idempotent)
    if output_db.get(item_id):
        return output_db[item_id]  # Return existing result

    # Process and store
    result = llm_process(restaurant)
    output_db[item_id] = result
    return result
```

**Real-world example:** Payment processing. If you charge a credit card and the response times out, you don't know if it went through. An idempotent API uses an `idempotency_key` — retrying with the same key won't charge twice.

---

## 5. All Interview Questions & Answers

### Prompt Engineering

**Q: How do you ensure an LLM outputs valid JSON consistently?**

A: Layer multiple strategies:
1. **System prompt:** "Output ONLY valid JSON, no markdown, no commentary"
2. **Few-shot example:** Show input → expected JSON output
3. **Tool use / function calling:** Force structured output via API schema
4. **Pydantic validation:** Catch anything that slips through
5. **Auto-repair loop:** Feed validation errors back to LLM to fix

The best approach is tool use (guaranteed by API), but when that's not available, the prompt + validate + repair loop is robust.

---

**Q: What's the difference between zero-shot, one-shot, and few-shot prompting?**

A:
- **Zero-shot:** "Extract restaurant data as JSON" — no examples
- **One-shot:** "Here's one example: [input] → [output]. Now do this: [new input]"
- **Few-shot:** "Here are 3 examples: ... Now do this: [new input]"

More examples = more consistent output format, but costs more tokens. For structured extraction, one-shot is usually the sweet spot.

---

**Q: How would you handle LLM hallucination in a data extraction pipeline?**

A:
1. **Constrain the output:** Use Pydantic with strict validators (e.g., rating must be 0-5)
2. **Cross-reference:** Check extracted data against the source text (does the name actually appear in the paragraph?)
3. **Confidence scoring:** Ask the LLM to rate its confidence, flag low-confidence extractions
4. **Human-in-the-loop:** Route uncertain outputs to human reviewers
5. **Multiple passes:** Run the same text through the LLM 3 times, take the majority answer

---

### Validation & Reliability

**Q: Why use Pydantic instead of just `json.loads()`?**

A: `json.loads()` only validates JSON syntax. Pydantic validates:
- **Types:** Is `rating` a float, not a string?
- **Required fields:** Is `name` present?
- **Constraints:** Is `rating` between 0 and 5?
- **Defaults:** Fill missing optional fields with defaults
- **Coercion:** Convert `"4.2"` string to `4.2` float automatically

For LLM outputs, Pydantic catches the subtle errors that `json.loads` misses.

---

**Q: Explain the auto-repair pattern — why not just retry the same prompt?**

A: Retrying the same prompt relies on randomness (temperature) to get a different result. Auto-repair is **directed** — you tell the LLM exactly what's wrong:

```
"Your JSON has error: 'rating' field is missing. Here's your output: {...}. Fix it."
```

This is like the difference between:
- Retry: "Try again" (student might make the same mistake)
- Auto-repair: "Your answer to Q3 is wrong because X. Fix Q3." (targeted correction)

Auto-repair has a ~90% success rate on first retry vs ~50% for blind retry.

---

**Q: How many retries are reasonable? What's the tradeoff?**

A: 3 retries (1 initial + 3 repairs) is the industry standard.

| Retries | Success Rate | Cost Multiplier | Time |
|---------|-------------|-----------------|------|
| 0 (no retry) | ~85% | 1x | 2s |
| 1 repair | ~95% | 2x | 4s |
| 2 repairs | ~98% | 3x | 6s |
| 3 repairs | ~99% | 4x | 8s |
| 10 repairs | ~99.1% | 11x | 22s |

Diminishing returns after 3. Beyond that, the input is probably genuinely ambiguous and needs human review.

---

### Multimodal

**Q: How do you send images to an LLM API?**

A: Two methods:
1. **Base64 encoding:** Convert image bytes to a base64 string, embed in the API request
2. **URL reference:** Pass a publicly accessible image URL (some APIs support this)

```python
# Base64 method (what we used)
with open("image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64
        }}
    ]
}
```

Media type detection matters — sending a PNG as `image/jpeg` can cause errors.

---

**Q: Why provide context (food name, review text) alongside the image?**

A: Without context, a vision LLM might describe: "A plate of food with red sauce and green garnish."

With context ("This is Classic Margherita Pizza"): "A traditional Neapolitan Margherita pizza featuring a blistered crust, San Marzano tomato sauce, fresh mozzarella, and vibrant basil leaves on a wooden board."

Context gives the model **grounding** — it knows what to focus on and uses domain-specific vocabulary. This produces captions that are more useful for downstream tasks like search and recommendation.

---

### System Design

**Q: How would you handle images that fail to download in a batch pipeline?**

A:
1. **Retry with backoff:** Network issues are usually transient (we used `tenacity`)
2. **Graceful degradation:** If all retries fail, save empty caption, don't skip the record
3. **Dead letter queue:** Log failed URLs for manual review
4. **Timeout:** Set a reasonable timeout (5-15s) so one slow image doesn't block the pipeline
5. **Parallel downloads:** Download images in batches, not one at a time

```python
recipe["image_description"] = caption if success else ""
# The record still exists in the dataset — just without a caption
# Better than losing the entire record
```

---

## 6. Concepts You Must Know

### Quick Reference Card

| Concept | One-Line Definition | Where We Used It |
|---------|-------------------|-----------------|
| **Prompt Engineering** | Designing LLM inputs to get desired outputs | System messages, few-shot examples |
| **Few-Shot Learning** | Teaching by example within the prompt | Example restaurant → JSON in prompt |
| **Schema Validation** | Checking data matches expected structure | Pydantic `model_validate_json()` |
| **Auto-Repair Loop** | Feed errors back to LLM for self-correction | Retry loop with validation errors |
| **Multimodal AI** | Models that process text + images + audio | Vision LLM for food image captioning |
| **Base64 Encoding** | Binary → text conversion for embedding in JSON | Image encoding for API calls |
| **CRUD Operations** | Create, Read, Update, Delete | CLI restaurant management |
| **Mocking** | Replacing real dependencies with fakes for testing | `unittest.mock.patch` for LLM calls |
| **Dependency Injection** | Passing dependencies from outside instead of hardcoding | Swappable LLM function |
| **Atomic Operations** | Operations that fully complete or don't happen at all | `os.replace()` for file writes |
| **Idempotency** | Same operation repeated = same result | Checkpoint-based processing |
| **Exponential Backoff** | Increasing wait time between retries | `tenacity` for image downloads |
| **Circuit Breaker** | Stop calling a failing service temporarily | Prevent hammering dead APIs |
| **Model Routing** | Use cheap models first, expensive only when needed | Haiku → Sonnet → Opus cascade |
| **Prompt Caching** | Cache repeated prompt prefixes to save cost | Anthropic `cache_control` |
| **Batch Processing** | Submit many requests at once, get results later | Anthropic Message Batches API |

---

### What Interviewers Actually Want to Hear

1. **You understand tradeoffs.** Not just "use Pydantic" but WHY Pydantic over json.loads, and WHEN you'd use something else.

2. **You think about failure modes.** "What if the API is down? What if the output is invalid? What if the disk is full?"

3. **You think about cost.** LLM calls cost money. Show you think about model selection, caching, and batching.

4. **You can scale.** "This works for 100 records. Here's how I'd make it work for 1M."

5. **You test properly.** You don't skip testing because "LLMs are non-deterministic." You mock, you validate, you eval.

---

*Generated for Module 1: Build a Structured Generative AI Application*
*IBM RAG and Agentic AI Capstone Project — April 2026*
