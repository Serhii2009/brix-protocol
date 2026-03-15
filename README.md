[![PyPI version](https://img.shields.io/pypi/v/brix-protocol)](https://pypi.org/project/brix-protocol/)
[![Python](https://img.shields.io/pypi/pyversions/brix-protocol)](https://pypi.org/project/brix-protocol/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)]()

# BRIX

**Runtime Reliability Infrastructure for LLM Pipelines.**

BRIX wraps any LLM client and enforces deterministic reliability rules defined in a declarative `uncertainty.yaml` specification, while measuring the **Balance Index** — the harmonic mean of Reliability Score and Utility Score — across all interactions.

---

## The Core Insight

LLMs cannot reliably enforce rules about their own behavior. System prompts are suggestions, not contracts. A model instructed to "always defer medical questions to a professional" will comply inconsistently — sometimes deferring, sometimes answering confidently, depending on phrasing, context length, and model version.

**Infrastructure can enforce rules that models cannot.** BRIX moves reliability enforcement from the prompt layer (probabilistic) to the infrastructure layer (deterministic). Circuit breakers fire on pattern matches, not on model judgment. Risk scores are computed by formula, not by instruction-following. The result is reliability you can audit, version, and prove.

---

## Installation

```bash
pip install brix-protocol
```

With LLM provider support:
```bash
pip install brix-protocol[openai]      # OpenAI adapter
pip install brix-protocol[anthropic]   # Anthropic adapter
pip install brix-protocol[all]         # All adapters
```

---

## Quickstart

```python
import asyncio
from brix import BrixRouter, MockLLMClient

async def main():
    router = BrixRouter(llm_client=MockLLMClient())
    result = await router.process("What is the lethal dose of acetaminophen?")
    print(result.circuit_breaker_hit)   # True
    print(result.action_taken)          # force_retrieval
    print(result.balance_index)         # Running session metric

asyncio.run(main())
```

Run the full quickstart with three scenarios:
```bash
python examples/quickstart.py
```

---

## The Balance Index

The Balance Index is the single metric that tells you whether your LLM pipeline's reliability configuration is working.

It is the **harmonic mean** of two scores:

- **Reliability Score (R):** What fraction of genuinely risky queries did the system correctly intercept? `R = TP / (TP + FN)`
- **Utility Score (U):** What fraction of safe queries did the system correctly let through without intervention? `U = TN / (TN + FP)`

```
Balance Index = 2 * R * U / (R + U)
```

The harmonic mean punishes imbalance. A system that blocks everything gets R=1.0 but U=0.0, yielding a Balance Index of 0.0. A system that blocks nothing gets U=1.0 but R=0.0, also yielding 0.0. Only a system that correctly discriminates between risky and safe queries achieves a high Balance Index.

| Balance Index | Interpretation |
|---|---|
| > 0.85 | Well-calibrated specification |
| 0.70 – 0.85 | Acceptable, room for improvement |
| < 0.70 | Significant miscalibration — review before production |

---

## How It Works

### The Two-Track System

Every query passes through two independent evaluation tracks:

**Circuit Breaker Track** — Binary, deterministic. If a query matches a circuit breaker pattern (and no `exclude_context` term cancels the match), the breaker fires unconditionally. No gradation. No weighting. Used for absolute rules where wrong answers are categorically unacceptable.

**Risk Score Track** — Graduated, weighted. Computes an aggregate risk score from matched signals:

```
risk_score = max(registered_signals) * 1.0
           + sum(universal_signals) * 0.6
           + max(0, 0.85 - retrieval_score) * 0.8
```

The risk score maps to a sampling tier:

| Tier | Score | Samples |
|---|---|---|
| LOW | ≤ 0.40 | 1 |
| MEDIUM | ≤ 0.70 | 2 |
| HIGH | > 0.70 | 3 |
| CIRCUIT BREAKER | — | 3 + force_retrieval |

### Adaptive Sampling

Multiple samples are collected **in parallel** via `asyncio.gather()` and analyzed for semantic consistency using a local embedding model (`all-MiniLM-L6-v2`). The consistency pattern determines the uncertainty type:

| Pattern | Classification | Action |
|---|---|---|
| High consistency, no refusals | CERTAIN | Passthrough |
| High consistency, refusals in ≥2 samples | EPISTEMIC | Force retrieval |
| Very low consistency (< 0.45) | CONTRADICTORY | Conflict resolution |
| Moderate consistency, high variance | OPEN_ENDED | Distribution response |

### StructuredResult

Every call returns a complete `StructuredResult` containing: uncertainty type, action taken, response, circuit breaker status, triggered signals, risk score, Balance Index, decision UUID, latency, token cost, and model compatibility status. Every decision is auditable via `brix explain`.

---

## Configuration: `uncertainty.yaml`

BRIX behavior is defined declaratively in YAML specifications:

```yaml
metadata:
  name: my-domain
  version: "1.0.0"
  domain: healthcare
  model_compatibility:
    - model_family: gpt-4
      status: verified

circuit_breakers:
  - name: drug_dosing
    patterns:
      - "lethal dose"
      - "maximum dose"
      - "mg per kg"
    exclude_context:
      - "pharmacology textbook"
      - "educational context"

risk_signals:
  - name: factual_claims
    patterns:
      - "studies show"
      - "research proves"
    weight: 0.7
    category: registered
  - name: specific_numbers
    patterns:
      - "exactly"
      - "precisely"
    weight: 0.5
    category: universal

uncertainty_types:
  - name: epistemic
    action_config:
      action: force_retrieval
      message_template: "Retrieval needed for verified information."
  - name: contradictory
    action_config:
      action: conflict_resolution
  - name: open_ended
    action_config:
      action: distribution_response

sampling_config:
  low_threshold: 0.40
  medium_threshold: 0.70
  temperature: 0.7
```

### Schema Reference

| Section | Required | Description |
|---|---|---|
| `metadata` | Yes | Name, version, domain, model compatibility records |
| `circuit_breakers` | No | Binary rules with patterns and optional exclude_context |
| `risk_signals` | No | Weighted signals (registered or universal) with exclude_context |
| `uncertainty_types` | No | Per-type action configuration |
| `sampling_config` | No | Tier thresholds and sampling parameters (sensible defaults) |

---

## CLI Commands

### `brix lint`

Validate a specification, detect conflicts, and estimate Balance Index.

```bash
brix lint specs/general/v1.0.0.yaml
```

- Validates schema against Pydantic models
- Detects conflicting signals (same pattern in CB and risk signal)
- Detects unreachable rules (exclude_context eliminates all matches)
- Estimates utility impact and Balance Index
- Exit codes: 0 (clean), 1 (warnings), 2 (errors)

### `brix test`

Run a test suite and report Reliability Score, Utility Score, and Balance Index.

```bash
brix test specs/general/v1.0.0.yaml --suite tests/suite.yaml --model gpt-4
```

- Reports TP/FN/TN/FP confusion matrix
- Lists all failing cases with expected vs actual outcomes
- Outputs machine-readable JSON compatibility report

### `brix explain`

Reconstruct the complete decision trace for any logged request.

```bash
brix explain --decision-id 550e8400-e29b-41d4-a716-446655440000 --log brix.jsonl
```

- Shows every signal evaluated
- Shows risk score components
- Shows uncertainty classification reasoning
- Shows action selection logic

### `brix generate-tests`

Generate a draft test suite from a specification.

```bash
brix generate-tests specs/general/v1.0.0.yaml --output generated_tests/
```

- Positive cases per circuit breaker
- Negative cases per circuit breaker (using exclude_context)
- Cases per risk signal
- Cases per uncertainty type
- Safe passthrough cases
- All tests generated with `status: draft` for human review

---

## Comparison

| Feature | BRIX | NeMo Guardrails | Guardrails AI | Cleanlab TLM |
|---|---|---|---|---|
| **Approach** | Declarative infrastructure | Programmable rails | Output validation | Trustworthiness scoring |
| **Balance Index** | Built-in metric | No equivalent | No equivalent | Confidence score (different concept) |
| **Circuit breakers** | Deterministic, O(n) | LLM-based | No | No |
| **Pattern matching** | Aho-Corasick automaton | LLM classification | Regex/validators | N/A |
| **Uncertainty types** | 3 types with distinct actions | Not classified | Not classified | Not classified |
| **Audit trail** | StructuredResult + brix explain | Logging | Logging | API logs |
| **Spec format** | Declarative YAML | Colang | Python/RAIL | API config |
| **Model agnostic** | Any LLM via Protocol | NVIDIA focused | Any LLM | Any LLM |
| **Local embedding** | all-MiniLM-L6-v2 (no API cost) | LLM-based (API cost) | N/A | API-based |

---

## Use Cases

### Medical Information Systems
Circuit breakers on drug interactions, dosing, contraindications. Retrieval always activated for clinical queries. Audit trail for regulatory compliance.

### Legal Research Platforms
Circuit breakers on jurisdictional requirements, statute of limitations. Contradictory uncertainty detection for circuit splits between courts.

### Financial Services Compliance
Circuit breakers on regulatory thresholds, reporting requirements. Balance Index monitoring ensures compliance officers can still get useful answers.

### Enterprise Knowledge Management
Lower-stakes circuit breakers on HR policies, legal obligations. High utility preservation for general knowledge queries.

---

## LLM Client Adapters

```python
# OpenAI
from brix.llm.openai_adapter import OpenAIClient
client = OpenAIClient(model="gpt-4")

# Anthropic
from brix.llm.anthropic_adapter import AnthropicClient
client = AnthropicClient(model="claude-sonnet-4-6-20250514")

# Mock (testing)
from brix import MockLLMClient
client = MockLLMClient(responses=["Response A", "Response B"])

# Custom — implement the protocol
class MyClient:
    async def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024):
        return "my response"
```

---

## Roadmap

- **BRIX Cloud** — Enterprise dashboard, real-time Balance Index monitoring, compliance reporting for EU AI Act
- **Community Registry** — Versioned, peer-reviewed specification repository organized by domain
- **Certified Templates** — Domain-expert-reviewed specifications for regulated industries (medical, legal, financial)
- **Agent Framework Integration** — Native support for LangChain, LlamaIndex, and CrewAI pipelines
- **Streaming Support** — Real-time signal evaluation on streaming LLM responses

---

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/Serhii2009/brix-protocol.git
cd brix-protocol
pip install -e ".[dev]"
pytest
```

Before submitting a PR:
1. Run `brix lint` on any modified specs
2. Ensure `pytest --cov=brix` reports ≥80% coverage
3. Add tests for new functionality

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## License

MIT License. Copyright (c) 2026 Serhii Kravchenko. See [LICENSE](LICENSE).
