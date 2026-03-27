<div align="center">

# BRIX — Runtime Reliability Infrastructure for LLM Pipelines

_One wrap() call. Seven guards. Zero hidden coupling._

[![PyPI version](https://img.shields.io/pypi/v/brix-protocol?cachebust=0)](https://pypi.org/project/brix-protocol/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)]()

</div>

---

BRIX wraps any LLM client with a configurable chain of guards, each solving exactly one production failure mode. System prompts are suggestions; BRIX guards are contracts. Activate what you need via `BRIX.wrap()` parameters — everything else stays out of the way.

---

## Installation

```bash
pip install brix-protocol                   # core guards only
pip install "brix-protocol[regulated]"      # + regulated-domain analysis (~500 MB)
pip install "brix-protocol[openai]"         # + OpenAI adapter
pip install "brix-protocol[anthropic]"      # + Anthropic adapter
pip install "brix-protocol[all]"            # everything
```

---

## Quick Start

```python
import asyncio
import openai
from pydantic import BaseModel
from brix import BRIX


class Answer(BaseModel):
    response: str
    confidence: float


async def main():
    client = BRIX.wrap(
        openai.AsyncOpenAI(),
        max_cost_usd=1.0,           # BudgetGuard: hard cost cap
        requests_per_minute=60,     # RateLimitGuard: adaptive throttle
        per_call_timeout=10.0,      # TimeoutGuard: 10 s per call
        max_retries=3,              # RetryGuard: 429/5xx → backoff
        response_schema=Answer,     # SchemaGuard: guaranteed structured output
        log_path="./traces",        # ObservabilityGuard: audit log + replay
    )

    result = await client.complete(
        [{"role": "user", "content": "What is the capital of France?"}],
        model="gpt-4o-mini",
    )
    print(result.response)    # "Paris"
    print(result.confidence)  # 0.99

    traces = client.get_traces()
    print(traces[0]["latency_ms"], traces[0]["chain_hash"])


asyncio.run(main())
```

---

## The Guard System

| Guard | Activated by | Guarantee |
|-------|-------------|-----------|
| **BudgetGuard** | `max_cost_usd` | No tokens spent after budget exhausted |
| **RateLimitGuard** | `requests_per_minute` | Average throughput ≤ configured rate |
| **TimeoutGuard** | `per_call/per_step/total_timeout` | `asyncio.wait_for` absolute — no call outlives limit |
| **ObservabilityGuard** | _always active_ | Every call in buffer; SHA-256 chained audit log |
| **SchemaGuard** | `response_schema` | Return value is valid Pydantic instance or raises |
| **RetryGuard** | `max_retries` | Transient errors retried with jittered exponential backoff |
| **RegulatedGuard** | `regulated_spec` | Circuit breakers + risk scoring for regulated domains |

**Execution order:** `Budget → RateLimit → Timeout → Observability → Schema → Retry → Regulated`

---

## Guard Reference

### BudgetGuard

Estimates prompt cost with tiktoken _before_ each call. Blocks (or warns) if the estimated total would exceed the session budget. Updates the actual cost from response usage tokens in `post_call`.

- `max_cost_usd` — session cost cap in USD
- `budget_strategy="block"` — `"block"` raises; `"warn"` emits a UserWarning and continues
- `budget_warning_threshold=0.8` — warn when 80% of budget is consumed

**Guarantee:** A blocked call consumes zero LLM tokens.

Raises: `BrixBudgetError`

---

### RateLimitGuard

Adaptive token bucket that throttles calls to stay under `requests_per_minute`. Detects 429 responses (via RetryGuard's `retry_history`) and halves the effective rate; recovers multiplicatively once no 429 occurs for `recovery_window_seconds`.

- `requests_per_minute` — target average rate cap
- `adaptive_rate_limiting=True` — auto-adjust on 429
- `burst_capacity=None` — optional hard cap on token bucket size (limits short bursts independently of average rate)
- `rate_reduction_factor=0.5` — multiply effective rate by this on each 429
- `rate_recovery_factor=1.05` — multiply effective rate by this on recovery (raise to 1.5 for faster recovery)
- `recovery_window_seconds=60.0` — seconds without a 429 before recovery fires

**Guarantee:** Average throughput cannot exceed configured rate. Provider burst-window limits (e.g. max N requests in 10 s) may still fire when `burst_capacity` is unset.

Raises: nothing — sleeps non-blockingly until a token is available.

---

### TimeoutGuard

Enforces three independent timeout levels at different abstraction layers.

- `per_call_timeout` — max wall-clock seconds for a single LLM call (via `asyncio.wait_for`)
- `per_step_timeout` — max seconds between the start of consecutive calls
- `total_timeout` — max seconds for the entire session (measured from `context.session_start`)
- `on_timeout="raise"` — `"raise"` raises `BrixTimeoutError`; `"return_partial"` returns an empty response and continues

**Guarantee:** `asyncio.wait_for` is absolute — no single call can outlive `per_call_timeout`.

Raises: `BrixTimeoutError`

---

### ObservabilityGuard _(always active)_

Records every call to an in-memory circular buffer. When `log_path` is set, also writes a JSONL audit log (`brix_audit.jsonl`) and per-session DRE files (`.brix_sessions/{session_id}.jsonl`) for deterministic replay.

- `log_path=None` — buffer-only mode; set to a directory to enable disk writes
- `trace_buffer_size=1000` — max in-memory trace entries (oldest evicted)
- `strict_mode=False` — when `True`, disk write failures raise instead of logging at WARNING
- `max_session_records=None` — rotate the DRE session file after this many records

**Guarantee:** Every completed call is added to the in-memory buffer. Disk writes are best-effort (`strict_mode=False`) or fail-fast (`strict_mode=True`). Replay requires successful disk writes.

Raises: `BrixGuardError` (only when `strict_mode=True`)

```python
traces = client.get_traces()   # list[dict], most-recent first
traces[0]["latency_ms"]
traces[0]["chain_hash"]        # SHA-256 of previous entry
traces[0]["prompt_hash"]       # SHA-256(json.dumps(messages))
```

---

### SchemaGuard

Injects the Pydantic model's JSON schema into the system prompt before the call (idempotent). After the call, extracts JSON from the response (handles markdown code fences), validates it, and returns a typed model instance. On failure, re-prompts with field-specific error messages up to `max_schema_retries` times.

- `response_schema` — Pydantic model class
- `max_schema_retries=2` — max self-healing re-prompt attempts (total attempts = retries + 1)
- `inject_schema_in_prompt=True` — inject JSON schema into system prompt in `pre_call`
- `max_healing_seconds=None` — optional wall-clock budget for the entire healing loop

**Guarantee:** If `complete()` returns without raising, the return value is a valid instance of `response_schema`. A partially validated or unvalidated object cannot be returned.

Raises: `BrixSchemaError` (with full attempt history)

---

### RetryGuard

Short-circuit guard — calls the LLM itself and retries on transient failures. The chain's own LLM call is skipped. Stores the full retry history in `context.metadata["retry_history"]` for RateLimitGuard's adaptive logic.

- `max_retries` — max retry attempts (total attempts = max_retries + 1)
- `backoff_base=2.0` — exponential backoff base
- `max_backoff=60.0` — max delay between retries in seconds
- `retry_budget_seconds=120.0` — total time budget for all retry delays
- `retry_on=None` — additional HTTP status codes to treat as retryable

**Guarantee:** Fatal errors (auth, invalid request body, etc.) are never retried. Retry history is always available in `context.metadata["retry_history"]` after the call.

Raises: `BrixGuardError` (retry budget exhausted)

---

### RegulatedGuard

Short-circuit guard — runs BrixRouter's two-track evaluation (deterministic circuit breakers + semantic risk scoring) on every call. The full result is stored in `context.metadata["regulated_result"]`.

- `regulated_spec` — path to a YAML spec file, or a built-in name: `"medical"`, `"legal"`, `"finance"`, `"hr"`, `"general"`

Requires: `pip install "brix-protocol[regulated]"`

**Guarantee:** Circuit breakers fire deterministically on pattern matches, not on model judgment. A `BrixGuardBlockedError` is raised when mandatory intervention is required.

Raises: `BrixGuardBlockedError`

```python
result = client.context.metadata["regulated_result"]
print(result.circuit_breaker_hit)   # bool
print(result.action_taken)          # ActionTaken enum
print(result.balance_index)         # harmonic mean of reliability + utility
print(result.uncertainty_type)      # UncertaintyType enum
```

---

## Observability & Replay

### Audit Log

Each entry in `brix_audit.jsonl` contains:

| Field | Description |
|-------|-------------|
| `run_id` | UUID4, unique per call |
| `session_id` | Fixed per `BrixClient` instance |
| `sequence` | Call counter within session |
| `timestamp` | ISO 8601 UTC |
| `model` | Model identifier |
| `prompt_tokens` / `completion_tokens` | From response usage |
| `cost_usd` | From BudgetGuard metadata |
| `latency_ms` | Wall-clock time for the call |
| `prompt_hash` | SHA-256 of serialized messages |
| `response_hash` | SHA-256 of response content |
| `guards_active` | Names of active guards |
| `chain_hash` | SHA-256 of previous entry's canonical JSON |

The `chain_hash` forms a cryptographic chain: the first entry's hash covers `{"genesis": session_id}`; each subsequent hash covers the complete previous entry. Any modification or deletion invalidates all subsequent hashes.

### DRE — Deterministic Replay Engine

Session files are written to `.brix_sessions/{session_id}.jsonl` alongside the audit log. They store response content in serializable form (`str` or Pydantic `model_dump()`) with enough metadata to reconstruct the original typed objects.

```python
import openai
from pydantic import BaseModel
from brix import BRIX


class Answer(BaseModel):
    response: str
    confidence: float


async def example():
    # --- Recording ---
    client = BRIX.wrap(
        openai.AsyncOpenAI(),
        log_path="./traces",
        response_schema=Answer,
    )
    session_id = client.context.session_id
    await client.complete([{"role": "user", "content": "Capital of France?"}])

    # --- Replay (zero LLM cost, no network) ---
    replay = BRIX.replay(
        session_id=session_id,
        log_path="./traces",
        schema=Answer,           # reconstructs Pydantic instance
    )
    print(replay.total_calls)    # 1
    result = await replay.complete()
    assert isinstance(result, Answer)

    # --- Housekeeping ---
    deleted = BRIX.purge_sessions("./traces", older_than_days=7)
    print(f"Purged {deleted} old session files")
```

`BrixReplayClient` properties: `session_id`, `total_calls`, `calls_remaining`.
`acomplete` is an alias for `complete`.

---

## Regulated Domain

For regulated industries (medical, legal, finance, HR), BRIX includes a two-track evaluation engine: deterministic **circuit breakers** (pattern matching, mandatory interventions) and probabilistic **risk scoring** (semantic consistency, uncertainty classification).

### Via `BRIX.wrap()`

```python
from brix import BRIX
import openai

client = BRIX.wrap(
    openai.AsyncOpenAI(),
    regulated_spec="medical",   # activates RegulatedGuard
)
await client.complete([{"role": "user", "content": "What is the max safe aspirin dose?"}])

result = client.context.metadata["regulated_result"]
print(result.circuit_breaker_hit)
print(result.action_taken)
print(result.balance_index)
```

### Via `BrixRouter` (standalone)

```python
from brix.regulated import BrixRouter, MockLLMClient

router = BrixRouter(llm_client=MockLLMClient(), spec="medical")
result = await router.process("What is the lethal dose of acetaminophen?")
print(result.circuit_breaker_hit)   # True
print(result.action_taken)          # force_retrieval
print(result.balance_index)         # running session metric
```

### Built-in Specs

| Name | Domain |
|------|--------|
| `"medical"` | FDA-aligned medical/clinical |
| `"legal"` | Legal research and advice |
| `"finance"` | Financial services |
| `"hr"` | Human resources |
| `"general"` | General-purpose (default) |

Load custom specs:

```python
from brix.regulated import load_spec, load_spec_from_dict

spec = load_spec("path/to/my_spec.yaml")
spec = load_spec_from_dict({"name": "custom", ...})
```

### RAG Integration

```python
from brix import RetrievalProvider, RetrievalResult
from brix.regulated import BrixRouter

class MyRAG(RetrievalProvider):
    async def retrieve(self, query: str, *, max_results: int = 3) -> RetrievalResult:
        docs = await my_vector_db.search(query, limit=max_results)
        return RetrievalResult(
            content="\n".join(d.text for d in docs),
            score=docs[0].score,
            sources=[d.url for d in docs],
        )

router = BrixRouter(
    llm_client=my_llm,
    spec="medical",
    retrieval_provider=MyRAG(),
)
```

### Standalone Output Guard

```python
from brix import OutputGuard, load_spec

guard = OutputGuard(load_spec("my_spec.yaml"))
result = await guard.analyze(response_text, query=original_query)
```

---

## Exception Hierarchy

All exceptions inherit from `BrixError`.

| Exception | Raised by | Condition |
|-----------|-----------|-----------|
| `BrixBudgetError` | BudgetGuard | Cost limit exceeded |
| `BrixTimeoutError` | TimeoutGuard | Time limit exceeded |
| `BrixSchemaError` | SchemaGuard | Validation failed after all retries |
| `BrixGuardBlockedError` | RetryGuard, RegulatedGuard | Request blocked or all retries exhausted |
| `BrixGuardError` | ObservabilityGuard (`strict_mode=True`) | Guard-internal failure |
| `BrixReplayError` | `BrixReplayClient` | Missing session file or no recorded response |
| `BrixConfigurationError` | `BrixClient` | Unsupported or misconfigured LLM client |

---

## `BRIX.wrap()` Parameter Reference

### BudgetGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_cost_usd` | `float \| None` | `None` | Session cost cap; activates guard |
| `budget_strategy` | `str` | `"block"` | `"block"` raises; `"warn"` continues |
| `budget_warning_threshold` | `float` | `0.8` | Warn at this fraction of the limit |

### RateLimitGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_per_minute` | `int \| None` | `None` | Target rate cap; activates guard |
| `adaptive_rate_limiting` | `bool` | `True` | Auto-reduce rate on 429 |
| `min_rate_floor` | `float` | `0.1` | Effective rate never drops below `max × floor` |
| `rate_reduction_factor` | `float` | `0.5` | Multiply rate by this on each 429 |
| `rate_recovery_factor` | `float` | `1.05` | Multiply rate by this on recovery |
| `recovery_window_seconds` | `float` | `60.0` | Seconds without 429 before recovery |
| `burst_capacity` | `int \| None` | `None` | Hard cap on token bucket size |

Legacy alias: `rate_limit_rpm` → `requests_per_minute`

### TimeoutGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_call_timeout` | `float \| None` | `None` | Max seconds for single LLM call |
| `per_step_timeout` | `float \| None` | `None` | Max seconds between consecutive calls |
| `total_timeout` | `float \| None` | `None` | Max seconds for entire session |
| `on_timeout` | `str` | `"raise"` | `"raise"` or `"return_partial"` |

Legacy alias: `max_time_seconds` → `per_call_timeout`

### ObservabilityGuard _(always active)_

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_path` | `str \| Path \| None` | `None` | Audit log directory; `None` = buffer-only |
| `trace_buffer_size` | `int` | `1000` | Max in-memory trace entries |
| `strict_mode` | `bool` | `False` | Raise on disk write failure |
| `max_session_records` | `int \| None` | `None` | Rotate DRE file after N records |

### SchemaGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_schema` | `type \| None` | `None` | Pydantic model class; activates guard |
| `max_schema_retries` | `int` | `2` | Max self-healing re-prompt attempts |
| `inject_schema_in_prompt` | `bool` | `True` | Inject JSON schema in system prompt |
| `max_healing_seconds` | `float \| None` | `None` | Wall-clock budget for healing loop |

### RetryGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int \| None` | `None` | Max retry attempts; activates guard |
| `backoff_base` | `float` | `2.0` | Exponential backoff base |
| `max_backoff` | `float` | `60.0` | Max delay between retries (seconds) |
| `retry_budget_seconds` | `float` | `120.0` | Total time budget for all retry delays |
| `retry_on` | `list[int] \| None` | `None` | Extra HTTP status codes to retry |

### RegulatedGuard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regulated_spec` | `str \| Path \| None` | `None` | Spec path or built-in name; activates guard |

---

## CLI

The `brix` CLI provides tooling for regulated-domain spec management:

```bash
brix lint   spec.yaml    # validate spec syntax and circuit breaker rules
brix test   spec.yaml    # run the spec's built-in test cases
brix explain spec.yaml   # explain what each rule does in plain language
brix generate spec.yaml  # generate test cases from spec rules
```
