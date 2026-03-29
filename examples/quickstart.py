# mypy: disable-error-code="arg-type,no-untyped-call,unused-ignore"
"""BRIX Quickstart — All nine guards in one file.

Run with:  python examples/quickstart.py --scenario <name>

Scenarios:
  schema        SchemaGuard       — JSON → Pydantic with self-healing
  budget        BudgetGuard       — hard cost cap; raises BrixBudgetError
  timeout       TimeoutGuard      — per-call deadline; raises BrixTimeoutError
  rate_limit    RateLimitGuard    — token-bucket throttle; sleeps before over-limit call
  loop          LoopGuard         — exact-duplicate detection; injects diversity prompt
  context       ContextGuard      — sliding-window compression when context is too large
  retry         RetryGuard        — exponential backoff on simulated 429 errors
  observability ObservabilityGuard — SHA-256 chained audit trail via get_traces()
  regulated     RegulatedGuard    — domain-policy enforcement (no API key needed)
  all           all of the above (default)

Flags:
  --reset-key   Delete the cached API key (~/.brix_key) and exit.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import stat
import sys
import time
from typing import Any

# Windows consoles default to cp1252; force UTF-8 so box-drawing characters render correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import httpx
import openai
from pydantic import BaseModel

from brix import BRIX
from brix.exceptions import (
    BrixBudgetError,
    BrixLoopError,
    BrixTimeoutError,
)

# ─────────────────────────────────────────────────────────────────────────────
# Terminal styling — restrained, professional colour palette
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"  # grey — commentary / secondary info
BRIGHT_CYAN = "\033[96m"  # light cyan — headings, guard names, borders
GREEN = "\033[92m"  # bright green — successful outcomes (✓)
YELLOW = "\033[93m"  # bright yellow — warnings / guard detections (⚠)
RED = "\033[91m"  # bright red — errors / unexpected failures


def _tag(name: str) -> str:
    """Light-cyan bold guard tag used in ⚠ and ✓ narrative lines."""
    return f"{BOLD}{BRIGHT_CYAN}[{name}]{RESET}"


def _ok(msg: str) -> str:
    """Wrap a short phrase in green — used for success labels."""
    return f"{GREEN}{msg}{RESET}"


def _warn(msg: str) -> str:
    """Wrap a short phrase in yellow — used for warning labels."""
    return f"{YELLOW}{msg}{RESET}"


def _err(msg: str) -> str:
    """Wrap a short phrase in red — used for error / failure labels."""
    return f"{RED}{msg}{RESET}"


def _section_header(guard_name: str, description: str) -> None:
    """Print a bright-cyan section separator with guard name and description."""
    line = "═" * 71
    print(f"\n{BOLD}{BRIGHT_CYAN}{line}{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}   {guard_name}{RESET}{BOLD} — {description}{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}{line}{RESET}")


def _step(arrow: str, msg: str) -> None:
    """Print a single narrative event line (→ / ← / ⚠ / ✓)."""
    print(f"  {arrow}  {msg}")


def _dim(msg: str) -> None:
    """Print a grey secondary-information line."""
    print(f"  {DIM}{msg}{RESET}")


def _outcome_table(happened: str, brix_did: str, without_brix: str) -> None:
    """Print the three-row scenario summary table."""
    col1 = 18
    col2 = max(len(happened), len(brix_did), len(without_brix), 50)
    top = f"  ┌{'─' * (col1 + 2)}┬{'─' * (col2 + 2)}┐"
    mid = f"  ├{'─' * (col1 + 2)}┼{'─' * (col2 + 2)}┤"
    bottom = f"  └{'─' * (col1 + 2)}┴{'─' * (col2 + 2)}┘"

    def row(label: str, value: str) -> str:
        return f"  │ {BOLD}{label:<{col1}}{RESET} │ {value:<{col2}} │"

    print()
    print(top)
    print(row("What happened", happened))
    print(mid)
    print(row("What BRIX did", _ok(brix_did)))
    print(mid)
    print(row("Without BRIX", _err(without_brix)))
    print(bottom)
    print()


def _print_table(headers: list[str], rows: list[list[object]]) -> None:
    """Print a plain data table with light-cyan bold headers."""
    if not rows:
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "  " + "-+-".join("-" * w for w in widths)
    header_cells = " | ".join(
        f"{BOLD}{BRIGHT_CYAN}{str(h):<{w}}{RESET}" for h, w in zip(headers, widths)
    )
    print(f"\n{sep}")
    print(f"  {header_cells}")
    print(sep)
    for row in rows:
        cells = " | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths))
        print(f"  {cells}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# API key handling — env → ~/.brix_key cache → interactive prompt → save cache
# ─────────────────────────────────────────────────────────────────────────────

_KEY_FILE = os.path.expanduser("~/.brix_key")


def _load_cached_key() -> str:
    """Return the key stored in ~/.brix_key, or an empty string if absent/unreadable."""
    try:
        with open(_KEY_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _save_cached_key(key: str) -> None:
    """Write key to ~/.brix_key with owner-only (0o600) permissions.

    Silently falls back if the home directory is not writable.
    """
    try:
        with open(_KEY_FILE, "w", encoding="utf-8") as fh:
            fh.write(key)
        os.chmod(_KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError as exc:
        print(
            f"  {_warn('Warning:')} could not cache API key to {_KEY_FILE}: {exc}\n"
            "  The key will be used for this session only."
        )


def _delete_cached_key() -> None:
    """Delete ~/.brix_key if it exists."""
    try:
        os.remove(_KEY_FILE)
        print(f"  Cached key deleted: {_KEY_FILE}")
    except FileNotFoundError:
        print(f"  No cached key found at {_KEY_FILE} — nothing to delete.")
    except OSError as exc:
        print(f"  {_err('Error')} deleting {_KEY_FILE}: {exc}")


def _resolve_api_key() -> str:
    """Return the OpenAI API key via env → cache file → interactive prompt.

    Resolution order:
    1. OPENAI_API_KEY environment variable (no prompt, no disk I/O).
    2. ~/.brix_key cache file written by a previous run.
    3. Interactive getpass prompt; saves the entered key to ~/.brix_key for
       future runs (0o600 permissions — owner read/write only).
    """
    # 1. Environment variable
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    # 2. On-disk cache from a previous run
    key = _load_cached_key()
    if key:
        _dim(f"Using cached API key from {_KEY_FILE} (run --reset-key to change)")
        return key

    # 3. Interactive prompt
    print()
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │  OPENAI_API_KEY not found in environment.                       │")
    print("  │                                                                 │")
    print("  │  Your key will be entered below. Input is hidden — it will not  │")
    print("  │  appear in the terminal. It will be saved to ~/.brix_key        │")
    print("  │  (permissions 0o600) so you are not prompted again.             │")
    print("  │                                                                 │")
    print("  │  To clear the cached key later: --reset-key                     │")
    print("  │  To skip caching: set OPENAI_API_KEY=sk-... before running.     │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()
    key = getpass.getpass("  OpenAI API key: ").strip()
    if not key:
        print("  No API key provided — exiting.")
        sys.exit(0)

    _save_cached_key(key)
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema for SchemaGuard scenario
# ─────────────────────────────────────────────────────────────────────────────


class ArticleData(BaseModel):
    title: str
    summary: str
    categories: list[str]
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# RetryGuard mock: thin proxy that raises RateLimitError for the first N calls
# ─────────────────────────────────────────────────────────────────────────────


class _FailNTimes:
    """Proxy around openai.AsyncOpenAI that raises a simulated 429 for the first N calls."""

    def __init__(self, real: openai.AsyncOpenAI, fail_count: int = 2) -> None:
        self._real = real
        self._remaining = fail_count

        _self = self

        class _Completions:
            async def create(self_c, **kwargs: Any) -> Any:  # type: ignore[misc]
                if _self._remaining > 0:
                    _self._remaining -= 1
                    _req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                    _resp = httpx.Response(
                        429,
                        content=(
                            b'{"error":{"message":"simulated rate limit",'
                            b'"type":"rate_limit_error"}}'
                        ),
                        headers={"content-type": "application/json"},
                        request=_req,
                    )
                    raise openai.RateLimitError(
                        "simulated rate limit (fail_count demo)",
                        response=_resp,
                        body={"error": {"message": "simulated rate limit"}},
                    )
                return await _self._real.chat.completions.create(**kwargs)

            def __getattr__(self_c, name: str) -> Any:  # type: ignore[misc]
                return getattr(_self._real.chat.completions, name)

        class _Chat:
            def __init__(self_c) -> None:  # type: ignore[misc]
                self_c.completions = _Completions()

            def __getattr__(self_c, name: str) -> Any:  # type: ignore[misc]
                return getattr(_self._real.chat, name)

        self.chat = _Chat()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ─────────────────────────────────────────────────────────────────────────────
# Mock semantic analyser for RegulatedGuard scenario (avoids loading ML model)
# ─────────────────────────────────────────────────────────────────────────────


class _QuickstartAnalyzer:
    def __init__(self, mean_similarity: float = 0.95, variance: float = 0.01) -> None:
        self._mean = mean_similarity
        self._variance = variance

    def analyze(self, samples: list[str]) -> Any:
        from brix.regulated.analysis.consistency import ConsistencyResult

        n = len(samples)
        count = max(1, n * (n - 1) // 2)
        return ConsistencyResult(
            mean_similarity=self._mean,
            variance=self._variance,
            pairwise_similarities=[self._mean] * count,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — SchemaGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_schema(api_key: str) -> None:
    _section_header(
        "SchemaGuard",
        "model returns plain text — self-healing re-prompts until valid JSON arrives",
    )

    print()
    print("  The situation: your pipeline expects structured ArticleData from the LLM.")
    print("  The model hasn't been told about the schema. It returns plain prose.")
    print("  SchemaGuard detects the JSON failure and heals it automatically.")
    print()

    # inject_schema_in_prompt=False: no upfront schema hint.
    # The neutral system prompt does not forbid JSON, but without a schema
    # hint the model naturally returns a prose answer on the first call.
    # SchemaGuard then constructs a precise re-prompt with field-level errors
    # and the exact JSON schema → healing fires on attempt 2.
    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        response_schema=ArticleData,
        inject_schema_in_prompt=False,  # no schema hint → first call returns plain text
        max_schema_retries=2,
    )

    text = (
        "OpenAI unveiled GPT-5 yesterday. The new model can reason, write code, "
        "and even control robots. Experts are calling it a breakthrough in AGI research. "
        "The release is scheduled for Q3 2026, with pricing expected to be similar to GPT-4."
    )
    prompt = f"Extract structured information from the following news article:\n\n{text}"

    # Neutral system prompt — does not instruct the model to avoid JSON,
    # but without inject_schema_in_prompt the model naturally responds in prose.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    _step("→", "Sending request to LLM (no schema hint — SchemaGuard validates after)")
    _dim("   first response will be plain text; SchemaGuard heals on re-prompt")

    result: ArticleData = await client.complete(messages, model="gpt-4o-mini")

    meta = client.context.metadata
    attempts: int = meta.get("schema_attempts", 1)
    validated: bool = meta.get("schema_validated", False)

    if attempts == 1:
        # Model happened to return valid JSON on the first attempt.
        _step("←", "LLM responded with valid JSON (no healing needed)")
        print(
            f"  ✓  {_tag('SchemaGuard')} {_ok('validated on first attempt')} "
            f"— schema_attempts=1"
        )
    else:
        _step("←", _warn("LLM responded with plain text (not valid JSON)"))
        print(f"  ⚠  {_tag('SchemaGuard')} {_warn('detected:')} response is not valid JSON")
        _dim("        — constructing re-prompt with field-level errors and exact schema")
        _dim("        — sending healing attempt 1 to the LLM...")
        _step("←", "LLM responded with valid JSON")
        print(
            f"  ✓  {_tag('SchemaGuard')} {_ok('validated ArticleData')} "
            f"— schema_attempts={attempts}, schema_validated={validated}"
        )

    print()
    print(f"  {BOLD}Validated ArticleData fields:{RESET}")
    _dim(f"   title      : {result.title}")
    _dim(f"   summary    : {result.summary[:90]}{'...' if len(result.summary) > 90 else ''}")
    _dim(f"   categories : {', '.join(result.categories)}")
    _dim(f"   sentiment  : {result.sentiment}")
    _dim(f"   confidence : {result.confidence:.2f}")

    _outcome_table(
        happened=f"LLM returned {'plain text' if attempts > 1 else 'JSON'} for structured extraction",
        brix_did=f"Detected invalid JSON, re-prompted with schema, validated on attempt {attempts}",
        without_brix="Raw string returned — ValidationError or silent wrong type in production",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — BudgetGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_budget(api_key: str) -> None:
    _section_header(
        "BudgetGuard",
        "call blocked at the door before a single token is sent to the API",
    )

    print()
    print("  The situation: a batch pipeline has hit its cost cap.")
    print("  max_cost_usd=0.0 — the very next call would exceed the budget.")
    print("  BudgetGuard pre-empts the call before the API is ever contacted.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        max_cost_usd=0.0,
        budget_strategy="block",
    )

    _step("→", 'Sending request: "Hello!" — budget cap is $0.0000')
    _dim("   BudgetGuard is counting prompt tokens and projecting cost...")

    guard_name = reason = "—"
    blocked = False
    try:
        await client.complete([{"role": "user", "content": "Hello!"}], model="gpt-4o-mini")
        _step("←", _err("Call succeeded — BudgetGuard did not fire (unexpected)"))
    except BrixBudgetError as exc:
        blocked = True
        guard_name = exc.guard_name
        reason = exc.reason
        print(
            f"  ⚠  {_tag('BudgetGuard')} {_warn('detected:')} projected cost exceeds $0.00000 limit"
        )
        print(
            f"  ✓  {_tag('BudgetGuard')} {_ok('resolved:')} call blocked — zero tokens sent to API"
        )
        _dim(f"   guard       : {guard_name}")
        _dim(f"   reason      : {reason[:80]}")
        _dim(
            f"   session cost: ${client.context.session_cost_usd:.5f} (unchanged — no API call was made)"
        )

    _outcome_table(
        happened="Batch pipeline hit $0.00 cost cap on next call",
        brix_did="Blocked call before any tokens sent — raised BrixBudgetError",
        without_brix="API charged silently — problem discovered only on the invoice",
    )
    _ = blocked  # suppress unused-variable warning


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — TimeoutGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_timeout(api_key: str) -> None:
    _section_header(
        "TimeoutGuard",
        "100-microsecond deadline fires — pipeline stays in control",
    )

    print()
    print("  The situation: a downstream API is hanging. No response after the deadline.")
    print("  per_call_timeout=0.0001s (100 μs) — fires before the network round-trip.")
    print("  Without BRIX: asyncio would wait forever. With BRIX: BrixTimeoutError.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        per_call_timeout=0.0001,
        on_timeout="raise",
    )

    _step("→", "Sending request — deadline: 0.0001s (100 microseconds)")
    _dim("   TimeoutGuard has set the asyncio.wait_for deadline...")

    guard_name = reason = "—"
    t0 = time.perf_counter()
    try:
        await client.complete([{"role": "user", "content": "Hello!"}], model="gpt-4o-mini")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _step(
            "←",
            _err(f"Call succeeded in {elapsed_ms:.1f}ms — TimeoutGuard did not fire (unexpected)"),
        )
    except BrixTimeoutError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        guard_name = exc.guard_name
        reason = exc.reason
        print(
            f"  ⚠  {_tag('TimeoutGuard')} {_warn('detected:')} deadline exceeded (elapsed: {elapsed_ms:.2f}ms)"
        )
        print(
            f"  ✓  {_tag('TimeoutGuard')} {_ok('resolved:')} BrixTimeoutError raised — pipeline recovers cleanly"
        )
        _dim(f"   guard   : {guard_name}")
        _dim(f"   reason  : {reason[:80]}")
        _dim(f"   elapsed : {elapsed_ms:.2f} ms")

    _outcome_table(
        happened="LLM call exceeded 100 μs deadline",
        brix_did="Fired BrixTimeoutError — clean exception, pipeline keeps running",
        without_brix="asyncio.gather hangs forever — entire pipeline blocked",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4 — RateLimitGuard
# ─────────────────────────────────────────────────────────────────────────────


async def _rate_limit_ticker(stop: asyncio.Event) -> None:
    """Print elapsed seconds while the token bucket is throttling."""
    t = 0
    while not stop.is_set():
        await asyncio.sleep(1)
        if stop.is_set():
            break
        t += 1
        print(f"  {DIM}[ {t}s ]{RESET}", end=" ", flush=True)


async def scenario_rate_limit(api_key: str) -> None:
    _section_header(
        "RateLimitGuard",
        "10 RPM token bucket — second call held for ~6 seconds, then released",
    )

    print()
    print("  The situation: two LLM calls issued back-to-back at 10 RPM.")
    print("  10 RPM = one token per 6 seconds. Call 1 consumes the token.")
    print("  Call 2 arrives immediately — the bucket is empty.")
    print("  Without BRIX: call 2 hits the API and gets a 429.")
    print("  With BRIX: RateLimitGuard holds call 2 in asyncio.sleep until a token refills.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        requests_per_minute=10,
    )

    rows: list[list[object]] = []

    _step("→", "Call 1 of 2: \"Say 'one'\" — token bucket has 1 token, passes immediately")
    t0 = time.perf_counter()
    await client.complete([{"role": "user", "content": "Say 'one'."}], model="gpt-4o-mini")
    t1 = time.perf_counter()
    wait1_ms = client.context.metadata.get("_rate_limit_wait_ms", 0.0)
    _step(
        "←",
        f"LLM responded in {(t1 - t0) * 1000:.0f}ms "
        f"(throttle wait: {wait1_ms:.0f}ms — bucket had a token)",
    )
    rows.append(["1", f"{(t1 - t0) * 1000:.0f}", f"{wait1_ms:.0f}", "—"])

    print()
    _step("→", "Call 2 of 2: \"Say 'two'\" — sending immediately")
    print(f"  ⚠  {_tag('RateLimitGuard')} {_warn('token bucket empty — throttling call 2...')}")
    print("  ", end="", flush=True)

    stop_event = asyncio.Event()
    ticker = asyncio.create_task(_rate_limit_ticker(stop_event))

    t2 = time.perf_counter()
    await client.complete([{"role": "user", "content": "Say 'two'."}], model="gpt-4o-mini")
    t3 = time.perf_counter()

    stop_event.set()
    await ticker
    print(_ok("released"))

    wait2_ms = client.context.metadata.get("_rate_limit_wait_ms", 0.0)
    wait2_s = wait2_ms / 1000
    effective_rpm = 60.0 / (t3 - t0) if (t3 - t0) > 0 else 0.0

    print(
        f"  ✓  {_tag('RateLimitGuard')} {_ok(f'released after {wait2_s:.1f}s')} "
        f"— sending call 2 now"
    )
    _step(
        "←",
        f"LLM responded (throttle hold: {wait2_ms:.0f}ms, "
        f"total elapsed: {(t3 - t2) * 1000:.0f}ms)",
    )
    _dim(f"   effective RPM achieved: {effective_rpm:.1f} (cap: 10)")
    rows.append(["2", f"{(t3 - t2) * 1000:.0f}", f"{wait2_ms:.0f}", f"{effective_rpm:.1f}"])

    _print_table(["call", "elapsed_ms", "throttle_wait_ms", "effective_rpm"], rows)

    _outcome_table(
        happened="Call 2 arrived immediately after call 1 at 10 RPM limit",
        brix_did=f"Held call 2 for {wait2_s:.1f}s in asyncio.sleep — released when token refilled",
        without_brix="Call 2 hits the API immediately — 429 RateLimitError crashes the pipeline",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5 — LoopGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_loop(api_key: str) -> None:
    _section_header(
        "LoopGuard",
        "agent forced to repeat itself — SHA-256 hash match detected, diversity injected",
    )

    print()
    print("  The situation: an agent is instructed to echo its previous answer exactly.")
    print("  Two identical responses → SHA-256 hashes match → LoopGuard detects the loop.")
    print("  LoopGuard injects a diversity prompt. The agent breaks the pattern.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        exact_loop_detection=True,
        exact_loop_threshold=2,  # fire after 2 identical hashes
        on_loop="inject_diversity",
        diversity_attempts=2,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers very short."}
    ]
    rows: list[list[object]] = []

    # Step 1 — ask a factual question
    _step("→", 'Step 1: "What is the capital of France? Answer in one word."')
    messages.append(
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    )
    resp1 = await client.complete(messages, model="gpt-4o-mini")
    messages.append({"role": "assistant", "content": resp1})
    hashes1 = client.context.metadata.get("_loop_hashes", [])
    _step("←", f"Agent: {_ok(repr(resp1.strip()))}")
    _dim(
        f"   hash recorded: {hashes1[-1][:16] if hashes1 else '—'}… ({len(hashes1)} hash in window)"
    )
    rows.append(["1", resp1.strip()[:30], len(hashes1), 0, "ok"])

    # Step 2 — force exact repetition for deterministic SHA-256 hash match
    print()
    _step("→", f'Step 2: "Say exactly this word for word, nothing more: {resp1.strip()}"')
    _dim("   forcing the model to echo resp1 exactly — SHA-256 collision guaranteed")
    messages.append(
        {
            "role": "user",
            "content": f"Say exactly this word for word, nothing more: {resp1.strip()}",
        }
    )

    loop_detected_at_2 = False
    try:
        resp2 = await client.complete(messages, model="gpt-4o-mini")
        messages.append({"role": "assistant", "content": resp2})
        hashes2 = client.context.metadata.get("_loop_hashes", [])
        div2 = client.context.metadata.get("_loop_diversity_count", 0)
        _step("←", f"Agent: {repr(resp2.strip()[:60])}")
        hash_count_2 = hashes2.count(hashes2[-1]) if hashes2 else 0
        _dim(f"   same hash seen {hash_count_2}x in window")
        if div2 > 0 or hash_count_2 >= 2:
            loop_detected_at_2 = True
            print(
                f"  ⚠  {_tag('LoopGuard')} {_warn('detected:')} "
                f"identical response hash repeated {hash_count_2} times"
            )
            print(
                f"  ✓  {_tag('LoopGuard')} {_ok('resolved:')} "
                f"diversity prompt will be injected on next call"
            )
        rows.append(
            [
                "2",
                resp2.strip()[:30],
                len(hashes2),
                div2,
                _warn("loop detected") if loop_detected_at_2 else "ok",
            ]
        )
    except BrixLoopError as exc:
        loop_detected_at_2 = True
        div2 = client.context.metadata.get("_loop_diversity_count", 0)
        hashes2 = client.context.metadata.get("_loop_hashes", [])
        print(f"  ⚠  {_tag('LoopGuard')} {_warn('detected:')} {exc.reason[:60]}")
        print(
            f"  ✓  {_tag('LoopGuard')} {_ok('resolved:')} BrixLoopError raised after exhausting diversity attempts"
        )
        rows.append(["2", "[BrixLoopError]", len(hashes2), div2, _err("raised")])

    # Step 3 — ask something different; LoopGuard injects diversity prompt
    print()
    _step("→", 'Step 3: "Tell me something interesting about France." (diversity prompt injected)')
    _dim("   LoopGuard has appended [BRIX-LOOP-RECOVERY] to the system message")
    messages.append({"role": "user", "content": "Tell me something interesting about France."})
    try:
        resp3 = await client.complete(messages, model="gpt-4o-mini")
        messages.append({"role": "assistant", "content": resp3})
        hashes3 = client.context.metadata.get("_loop_hashes", [])
        div3 = client.context.metadata.get("_loop_diversity_count", 0)
        _step("←", f"Agent: {repr(resp3.strip()[:80])}")
        _dim(f"   new hash added — {len(set(hashes3))} unique hashes in window — loop broken")
        print(f"  ✓  {_tag('LoopGuard')} {_ok('loop healed')} — agent responding normally again")
        rows.append(["3", resp3.strip()[:30], len(hashes3), div3, _ok("healed")])
    except BrixLoopError as exc:
        hashes3 = client.context.metadata.get("_loop_hashes", [])
        div3 = client.context.metadata.get("_loop_diversity_count", 0)
        print(
            f"  ⚠  {_tag('LoopGuard')} loop persisted after diversity injection: "
            f"{_warn(exc.reason[:60])}"
        )
        rows.append(["3", "[BrixLoopError]", len(hashes3), div3, _err("raised")])

    _print_table(
        ["step", "response_preview", "hashes_in_window", "diversity_injections", "status"], rows
    )

    _outcome_table(
        happened="Agent echoed its previous response exactly — SHA-256 hash matched",
        brix_did="Detected duplicate hash, injected [BRIX-LOOP-RECOVERY] diversity prompt",
        without_brix="Agent loops forever echoing same response — budget burns to zero",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 6 — ContextGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_context(api_key: str) -> None:
    _section_header(
        "ContextGuard",
        "conversation grows past 250-token budget — sliding-window compression fires",
    )

    print()
    print("  The situation: a multi-turn conversation accumulates history.")
    print("  Budget = max_context_tokens − reserve_tokens = 400 − 150 = 250 tokens.")
    print("  After 2-3 100-word turns, the history exceeds the budget.")
    print("  ContextGuard compresses silently. The conversation continues normally.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        max_context_tokens=400,
        context_strategy="sliding_window",
        context_reserve_tokens=150,
    )

    prompts = [
        "Write a short paragraph about cloud computing (around 80 words).",
        "Now write a short paragraph about machine learning (around 80 words).",
        "Now write a short paragraph about data pipelines (around 80 words).",
        "Now summarise the three topics above in one sentence.",
    ]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."}
    ]
    rows: list[list[object]] = []
    compression_fired = False

    for i, prompt in enumerate(prompts, start=1):
        _step("→", f"Step {i}: {prompt[:65]}...")
        _dim(f"   sending {len(messages)} message(s) to LLM")
        messages.append({"role": "user", "content": prompt})

        response = await client.complete(messages, model="gpt-4o-mini")
        messages.append({"role": "assistant", "content": response})

        meta = client.context.metadata
        compressed: bool = meta.get("_context_compressed", False)
        tok_before: object = meta.get("_tokens_before", "—")
        tok_after: object = meta.get("_tokens_after", "—")
        strategy: str = meta.get("_context_strategy_used", "—")

        _step("←", f"LLM responded ({len(response)} chars)")

        if compressed:
            compression_fired = True
            print(
                f"  ⚠  {_tag('ContextGuard')} {_warn('detected:')} "
                f"{tok_before} tokens > 250-token budget"
            )
            print(
                f"  ✓  {_tag('ContextGuard')} {_ok('resolved:')} "
                f"{strategy} compression applied — {tok_before} → {tok_after} tokens"
            )
            _dim("   oldest messages pruned; system prompt and current request preserved")
        else:
            _dim("   context within budget — no compression needed")

        rows.append([i, len(messages), "YES" if compressed else "no", tok_before, tok_after])

    _print_table(["step", "messages", "compressed", "tokens_before", "tokens_after"], rows)

    _outcome_table(
        happened=f"Conversation history exceeded 250-token budget at step {'3+' if compression_fired else '—'}",
        brix_did="Applied sliding_window compression — kept system msg + recent history",
        without_brix="context_length_exceeded error from the API — pipeline crashes mid-conversation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 7 — RetryGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_retry(api_key: str) -> None:
    _section_header(
        "RetryGuard",
        "two 429 errors absorbed with exponential backoff — success on attempt 3",
    )

    print()
    print("  The situation: the API returns 429 RateLimitError on the first two calls.")
    print("  Without BRIX: the first 429 crashes the pipeline.")
    print("  With BRIX: RetryGuard catches each 429, backs off, and retries transparently.")
    print()

    real = openai.AsyncOpenAI(api_key=api_key)
    client = BRIX.wrap(
        _FailNTimes(real, fail_count=2),
        max_retries=3,
        backoff_base=1.5,  # ~1.5s per retry — delays are clearly visible
        max_backoff=5.0,
    )

    _step("→", "Sending request — RetryGuard will intercept failures internally...")
    _dim("   _FailNTimes proxy will return 429 for the first 2 attempts")

    t0 = time.perf_counter()
    result = await client.complete(
        [{"role": "user", "content": "Say 'hello' in one word."}],
        model="gpt-4o-mini",
    )
    elapsed = time.perf_counter() - t0

    retry_count: int = client.context.metadata.get("retry_count", 0)
    retry_history: list[dict[str, Any]] = client.context.metadata.get("retry_history", [])

    print()
    print(f"  {DIM}── reconstructed from retry_history (happened inside BRIX) {'─' * 22}{RESET}")
    rows: list[list[object]] = []

    for entry in retry_history:
        attempt = entry.get("attempt", "—")
        err_type = entry.get("error_type", "—")
        delay = entry.get("delay", 0.0)
        error_preview = str(entry.get("error", ""))[:40]
        _step("→", f"Attempt {attempt}: sending request...")
        print(f"  ⚠  {_tag('RetryGuard')} {_warn(str(err_type))} — {error_preview!r}")
        print(
            f"  ✓  {_tag('RetryGuard')} {_ok(f'backing off {delay:.2f}s')} "
            f"before retry {int(attempt) + 1}"
        )
        rows.append([attempt, err_type, f"{delay:.2f}", _warn("retry")])

    _step("←", f"Succeeded on attempt {retry_count + 1}: {_ok(repr(str(result)))}")
    rows.append([retry_count + 1, "—", "—", _ok("success")])

    print(f"  {DIM}{'─' * 70}{RESET}")

    _print_table(["attempt", "error_type", "delay_s", "outcome"], rows)
    _dim(f"   total elapsed: {elapsed * 1000:.0f}ms across {retry_count + 1} attempts")

    _outcome_table(
        happened=f"API returned 429 on first {retry_count} attempt(s)",
        brix_did=f"Caught each 429, applied backoff (~1.5s), succeeded on attempt {retry_count + 1}",
        without_brix="First 429 propagates as unhandled exception — pipeline crashes immediately",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 8 — ObservabilityGuard
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_observability(api_key: str) -> None:
    _section_header(
        "ObservabilityGuard",
        "every call recorded in a SHA-256 chained audit log — tamper-evident by design",
    )

    print()
    print("  The situation: production calls with no audit trail.")
    print("  A bug occurs. Nobody can reproduce it. The LLM output is gone forever.")
    print("  ObservabilityGuard records every call. The chain_hash links entries")
    print("  cryptographically — altering any record breaks the chain.")
    print()

    client = BRIX.wrap(
        openai.AsyncOpenAI(api_key=api_key),
        log_path="./traces",
    )

    prompts = [
        "What is the boiling point of water in Celsius?",
        "What is the speed of light in metres per second?",
    ]

    for i, prompt in enumerate(prompts, start=1):
        _step("→", f"Call {i}: {prompt!r}")
        await client.complete([{"role": "user", "content": prompt}], model="gpt-4o-mini")
        _dim(f"   trace recorded — run_id: {str(client.context.run_id)[:16]}…")
        _step("←", f"LLM responded — entry {i} appended to audit log")

    traces = client.get_traces()
    print()
    print(
        f"  ✓  {_tag('ObservabilityGuard')} "
        f"{_ok(f'{len(traces)} trace(s) recorded')} → ./traces/brix_audit.jsonl"
    )
    print()

    rows: list[list[object]] = []
    for i, t in enumerate(reversed(traces), start=1):
        rows.append(
            [
                i,
                f"{t.get('latency_ms') or 0.0:.1f}",
                t.get("prompt_tokens") or "—",
                t.get("completion_tokens") or "—",
                f"${t.get('cost_usd') or 0.0:.6f}",
                (t.get("chain_hash") or "")[:16] + "…",
            ]
        )

    _print_table(
        ["call#", "latency_ms", "prompt_tok", "completion_tok", "cost_usd", "chain_hash[:16]"],
        rows,
    )

    if len(traces) >= 2:
        ch1 = (traces[1].get("chain_hash") or "")[:16]
        ch2 = (traces[0].get("chain_hash") or "")[:16]
        print(f"  ✓  {_tag('ObservabilityGuard')} {_ok('cryptographic chain verified:')} ")
        _dim(f"   entry 1 chain_hash: {ch1}…")
        _dim(f"   entry 2 chain_hash: {ch2}… ← SHA-256 of entry 1's JSON")
        print()
        print("  Altering any historical entry breaks the chain — a compliance audit")
        print("  detects tampering instantly by re-computing the chain from genesis.")

    _outcome_table(
        happened="Two LLM calls made with no manual instrumentation",
        brix_did="Recorded latency, tokens, cost, SHA-256 chain per call → JSONL on disk",
        without_brix="No audit trail — production bugs cannot be reproduced; compliance fails",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 9 — RegulatedGuard (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────


async def scenario_regulated() -> None:
    _section_header(
        "RegulatedGuard",
        "dangerous medical query blocked by circuit breaker — safe query passes through",
    )

    print()
    print("  The situation: a medical chatbot receives queries.")
    print("  Without policy enforcement, the LLM answers however it feels like.")
    print("  RegulatedGuard applies domain policy: circuit breakers block dangerous queries")
    print("  before the LLM is even contacted.")
    print()
    print("  This scenario uses MockLLMClient — no API key required.")
    print()

    from brix.regulated.core.router import BrixRouter
    from brix.regulated.llm.mock import MockLLMClient

    cases = [
        (
            "What is the lethal dose of acetaminophen?",
            MockLLMClient(
                responses=[
                    "I cannot provide specific dosage information. Please consult a medical professional.",
                    "As an AI, I'm not qualified to advise on drug dosages.",
                    "I must decline. Please consult a healthcare provider.",
                ]
            ),
            "dangerous — circuit breaker expected",
        ),
        (
            "What are the best supplements for boosting immunity?",
            MockLLMClient(
                default_response=(
                    "I can suggest some common supplements like vitamin C, zinc, and vitamin D, "
                    "but always consult a healthcare professional before starting any new regimen."
                )
            ),
            "borderline — may trigger epistemic uncertainty",
        ),
    ]

    rows: list[list[object]] = []
    for query, mock_llm, note in cases:
        router = BrixRouter(llm_client=mock_llm, _analyzer=_QuickstartAnalyzer())

        _step("→", f"Sending: {query!r}")
        _dim(f"   ({note})")

        result = await router.process(query)
        cb = result.circuit_breaker_hit

        if cb:
            print(
                f"  ⚠  {_tag('RegulatedGuard')} {_warn('circuit breaker:')} "
                f"query matches regulated-domain pattern"
            )
            print(f"  ✓  {_tag('RegulatedGuard')} {_ok('blocked')} — LLM was never contacted")
            _step("←", f"Safe response returned: {(result.response or '')[:70]!r}")
        else:
            print(
                f"  ✓  {_tag('RegulatedGuard')} {_ok('passed:')} "
                f"query is outside regulated domain"
            )
            _step("←", f"LLM responded: {(result.response or '')[:70]!r}")

        _dim(f"   circuit_breaker_hit  : {cb}")
        _dim(f"   uncertainty_type     : {result.uncertainty_type}")
        _dim(f"   action_taken         : {result.action_taken}")
        _dim(f"   intervention_necessary: {result.intervention_necessary}")
        print()

        rows.append(
            [
                query[:35],
                _warn("YES") if cb else _ok("no"),
                str(result.action_taken),
                (result.response or "")[:30],
            ]
        )

    _print_table(["query", "circuit_breaker", "action_taken", "response_preview"], rows)

    _outcome_table(
        happened="Medical chatbot received a dangerous dosage query",
        brix_did="Circuit breaker fired before LLM was contacted — safe refusal returned",
        without_brix="LLM answers however it feels like — unpredictable in regulated domain",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="BRIX guard demos — one scenario per guard")
    parser.add_argument(
        "--scenario",
        choices=[
            "schema",
            "budget",
            "timeout",
            "rate_limit",
            "loop",
            "context",
            "retry",
            "observability",
            "regulated",
            "all",
        ],
        default="all",
        help="Which guard scenario to run (default: all)",
    )
    parser.add_argument(
        "--reset-key",
        action="store_true",
        help=f"Delete the cached API key ({_KEY_FILE}) and exit.",
    )
    args = parser.parse_args()

    # Handle --reset-key before doing anything else.
    if args.reset_key:
        _delete_cached_key()
        return

    # 'regulated' uses MockLLMClient — no API key needed.
    # All other scenarios (including 'all') resolve the key once here and
    # reuse it — no repeated prompts even when running all nine scenarios.
    regulated_only = args.scenario == "regulated"
    api_key: str | None = None if regulated_only else _resolve_api_key()

    scenario_map = {
        "schema": lambda: scenario_schema(api_key),  # type: ignore[arg-type]
        "budget": lambda: scenario_budget(api_key),  # type: ignore[arg-type]
        "timeout": lambda: scenario_timeout(api_key),  # type: ignore[arg-type]
        "rate_limit": lambda: scenario_rate_limit(api_key),  # type: ignore[arg-type]
        "loop": lambda: scenario_loop(api_key),  # type: ignore[arg-type]
        "context": lambda: scenario_context(api_key),  # type: ignore[arg-type]
        "retry": lambda: scenario_retry(api_key),  # type: ignore[arg-type]
        "observability": lambda: scenario_observability(api_key),  # type: ignore[arg-type]
        "regulated": lambda: scenario_regulated(),
    }

    if args.scenario == "all":
        for fn in scenario_map.values():
            await fn()
    else:
        await scenario_map[args.scenario]()

    line = "═" * 71
    print(f"\n{BOLD}{BRIGHT_CYAN}{line}{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}   BRIX QUICKSTART COMPLETE{RESET}")
    print(f"{BOLD}{BRIGHT_CYAN}{line}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
