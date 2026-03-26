<div align="center">

# BRIX — Balance-Reliability IndeX

**Runtime Reliability Infrastructure for LLM Pipelines**

_Enforce deterministic rules. Measure the Balance Index. Audit every decision._

[![PyPI version](https://img.shields.io/pypi/v/brix-protocol?cachebust=0)](https://pypi.org/project/brix-protocol/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)]()

</div>

---

BRIX wraps any LLM client and enforces deterministic reliability rules defined in a declarative `uncertainty.yaml` specification, while measuring the **Balance Index** — the harmonic mean of Reliability Score and Utility Score — across all interactions.

---

## The Core Insight

LLMs cannot reliably enforce rules about their own behavior. System prompts are suggestions, not contracts. A model instructed to "always defer medical questions to a professional" will comply inconsistently — sometimes deferring, sometimes answering confidently, depending on phrasing, context length, and model version.

**Infrastructure can enforce rules that models cannot.** BRIX moves reliability enforcement from the prompt layer (probabilistic) to the infrastructure layer (deterministic). Circuit breakers fire on pattern matches, not on model judgment. Risk scores are computed by formula, not by instruction-following. The result is reliability you can audit, version, and prove.

---

## Installation

### pip

```bash
pip install brix-protocol
```

With regulated-domain support (sentence-transformers, ~500 MB):

```bash
pip install "brix-protocol[regulated]"
```

With LLM provider support:

```bash
pip install "brix-protocol[openai]"      # OpenAI adapter
pip install "brix-protocol[anthropic]"   # Anthropic adapter
pip install "brix-protocol[all]"         # regulated + all adapters
```

### Poetry

```bash
poetry add brix-protocol

# With regulated extras
poetry add "brix-protocol[regulated]"

# With all extras
poetry add "brix-protocol[all]"
```

---

## Quickstart

```python
import asyncio
import openai
from brix import BRIX

async def main():
    client = BRIX.wrap(openai.OpenAI())
    response = await client.complete([
        {"role": "user", "content": "What is the recommended daily dose of vitamin C?"}
    ])
    print(response)

asyncio.run(main())
```

Run the full quickstart with three scenarios:

```bash
python examples/quickstart.py
```

### Regulated domains

For fintech, medtech, legal, and other regulated industries, activate `RegulatedGuard`
via `regulated_spec`. Requires `pip install "brix-protocol[regulated]"`.

```python
import asyncio
from brix import BRIX
from brix.regulated import MockLLMClient
from brix.regulated.spec.defaults import get_medical_spec_path

async def main():
    client = BRIX.wrap(
        MockLLMClient(),
        regulated_spec=get_medical_spec_path(),   # activates RegulatedGuard
    )
    response = await client.complete([
        {"role": "user", "content": "What is the lethal dose of acetaminophen?"}
    ])
    # context.metadata["regulated_result"] holds the full StructuredResult
    result = client.context.metadata["regulated_result"]
    print(result.circuit_breaker_hit)   # True
    print(result.action_taken)          # force_retrieval
    print(result.balance_index)         # Running session metric

asyncio.run(main())
```

Or use `BrixRouter` directly for full access to `StructuredResult` on every call:

```python
import asyncio
from brix.regulated import BrixRouter, MockLLMClient

async def main():
    router = BrixRouter(llm_client=MockLLMClient())
    result = await router.process("What is the lethal dose of acetaminophen?")
    print(result.circuit_breaker_hit)   # True
    print(result.action_taken)          # force_retrieval
    print(result.balance_index)         # Running session metric

asyncio.run(main())
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

| Balance Index | Interpretation                                        |
| ------------- | ----------------------------------------------------- |
| > 0.85        | Well-calibrated specification                         |
| 0.70 – 0.85   | Acceptable, room for improvement                      |
| < 0.70        | Significant miscalibration — review before production |

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

| Tier            | Score  | Samples             |
| --------------- | ------ | ------------------- |
| LOW             | ≤ 0.40 | 1                   |
| MEDIUM          | ≤ 0.70 | 2                   |
| HIGH            | > 0.70 | 3                   |
| CIRCUIT BREAKER | —      | 3 + force_retrieval |

### Adaptive Sampling

Multiple samples are collected **in parallel** via `asyncio.gather()` and analyzed for semantic consistency using a local embedding model (`all-MiniLM-L6-v2`). The consistency pattern determines the uncertainty type:

| Pattern                                  | Classification | Action                |
| ---------------------------------------- | -------------- | --------------------- |
| High consistency, no refusals            | CERTAIN        | Passthrough           |
| High consistency, refusals in ≥2 samples | EPISTEMIC      | Force retrieval       |
| Very low consistency (< 0.45)            | CONTRADICTORY  | Conflict resolution   |
| Moderate consistency, high variance      | OPEN_ENDED     | Distribution response |

### StructuredResult

Every call returns a complete `StructuredResult` containing: uncertainty type, action taken, response, circuit breaker status, triggered signals, risk score, Balance Index, decision UUID, latency, token cost, model compatibility status, `response_requires_verification` (whether the response needs external verification), `unverified_draft` (raw LLM output for internal processing), `retrieval_executed`/`retrieval_sources` (RAG status), `sampler_partial_failure` (degraded sampling), and `output_result` (output guard analysis). Every decision is auditable via `brix explain`.

---

## Configuration: `uncertainty.yaml`

BRIX behavior is defined declaratively in YAML specifications:

```yaml
metadata:
  name: my-domain
  version: '1.0.0'
  domain: healthcare
  model_compatibility:
    - model_family: gpt-4
      status: verified

circuit_breakers:
  - name: drug_dosing
    patterns:
      - 'lethal dose'
      - 'maximum dose'
      - 'mg per kg'
    exclude_context:
      - 'pharmacology textbook'
      - 'educational context'

risk_signals:
  - name: factual_claims
    patterns:
      - 'studies show'
      - 'research proves'
    weight: 0.7
    category: registered
  - name: specific_numbers
    patterns:
      - 'exactly'
      - 'precisely'
    weight: 0.5
    category: universal

uncertainty_types:
  - name: epistemic
    action_config:
      action: force_retrieval
      message_template: 'Retrieval needed for verified information.'
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
  max_tokens: 1024
```

### Schema Reference

| Section             | Required | Description                                                                |
| ------------------- | -------- | -------------------------------------------------------------------------- |
| `metadata`          | Yes      | Name, version, domain, model compatibility records                         |
| `circuit_breakers`  | No       | Binary rules with patterns and optional exclude_context                    |
| `risk_signals`      | No       | Weighted signals (registered or universal) with exclude_context            |
| `uncertainty_types` | No       | Per-type action configuration                                              |
| `sampling_config`   | No       | Tier thresholds, sampling parameters, and `max_tokens` (sensible defaults) |
| `output_signals`    | No       | Response-side signals with `signal_type: risk\|block` for output guard     |

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

| Feature               | BRIX                            | NeMo Guardrails      | Guardrails AI     | Cleanlab TLM                         |
| --------------------- | ------------------------------- | -------------------- | ----------------- | ------------------------------------ |
| **Approach**          | Declarative infrastructure      | Programmable rails   | Output validation | Trustworthiness scoring              |
| **Balance Index**     | Built-in metric                 | No equivalent        | No equivalent     | Confidence score (different concept) |
| **Circuit breakers**  | Deterministic, O(n)             | LLM-based            | No                | No                                   |
| **Pattern matching**  | Aho-Corasick automaton          | LLM classification   | Regex/validators  | N/A                                  |
| **Uncertainty types** | 3 types with distinct actions   | Not classified       | Not classified    | Not classified                       |
| **Audit trail**       | StructuredResult + brix explain | Logging              | Logging           | API logs                             |
| **Spec format**       | Declarative YAML                | Colang               | Python/RAIL       | API config                           |
| **Model agnostic**    | Any LLM via Protocol            | NVIDIA focused       | Any LLM           | Any LLM                              |
| **Local embedding**   | all-MiniLM-L6-v2 (no API cost)  | LLM-based (API cost) | N/A               | API-based                            |

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

## Built-in Specifications

BRIX ships with five ready-to-use domain specifications:

| Spec             | Domain                | Circuit Breakers | Risk Signals | Balance Index |
| ---------------- | --------------------- | ---------------- | ------------ | ------------- |
| `general/v1.0.0` | General purpose       | 3                | 7            | 0.873         |
| `medical/v1.0.0` | Medical / FDA-aligned | 6                | 8            | 0.884         |
| `legal/v1.0.0`   | Legal research        | 5                | 7            | 0.895         |
| `finance/v1.0.0` | Financial services    | 5                | 8            | 0.894         |
| `hr/v1.0.0`      | Human resources       | 4                | 6            | 0.889         |

Load any spec by path:

```python
from brix.regulated.spec.defaults import get_medical_spec_path
from brix.regulated import BrixRouter, load_spec

spec = load_spec(get_medical_spec_path())
router = BrixRouter(llm_client=client, spec=spec)
```

---

## LLM Client Adapters

```python
# OpenAI
from brix.regulated.llm.openai_adapter import OpenAIClient
client = OpenAIClient(model="gpt-4")

# Anthropic
from brix.regulated.llm.anthropic_adapter import AnthropicClient
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

## Output Guard

BRIX can scan LLM responses for risky content using the same Aho-Corasick infrastructure. Enable it on the router:

```python
router = BrixRouter(llm_client=client, spec=spec, enable_output_guard=True)
result = await router.process("What treatment do you recommend?")

if result.output_result and result.output_result.output_blocked:
    print("Response blocked:", result.output_result.output_block_signal)
```

Define output signals in your spec with `signal_type: block` (hard block) or `signal_type: risk` (scored):

```yaml
output_signals:
  - name: definitive_diagnosis
    patterns: ['you have', 'you are diagnosed']
    weight: 0.9
    signal_type: block
```

For standalone use outside BrixRouter:

```python
from brix import OutputGuard, load_spec

guard = OutputGuard(load_spec("my_spec.yaml"))
result = await guard.analyze(response_text, query=original_query)
```

---

## Retrieval Provider

When BRIX detects epistemic uncertainty, it can execute real RAG instead of just signaling `force_retrieval`. Implement the `RetrievalProvider` protocol:

```python
from brix import RetrievalProvider, RetrievalResult

class MyRAG:
    async def retrieve(self, query: str, *, max_results: int = 3) -> RetrievalResult:
        docs = await my_vector_db.search(query, limit=max_results)
        return RetrievalResult(
            content="\n".join(d.text for d in docs),
            score=docs[0].score,
            sources=[d.url for d in docs],
        )

router = BrixRouter(llm_client=client, retrieval_provider=MyRAG())
```

When retrieval executes, `result.retrieval_executed` is `True` and `result.retrieval_sources` contains the source list. If retrieval fails, BRIX falls back gracefully with `result.retrieval_failed = True`.

---

## Console Output

BRIX provides optional one-line terminal feedback for each request. It activates automatically in TTY terminals and can be controlled via environment variables:

```
BRIX_CONSOLE=1    # Force on (even in pipes)
BRIX_CONSOLE=0    # Force off (even in terminals)
BRIX_VERBOSE=1    # Show signals, retrieval status, decision ID
```

Example output:

```
  OK  Safe -- passed through         risk 0.04   balance 0.892   12ms
  !   Elevated -- retrieval needed   risk 0.73   balance 0.871   87ms
  X   Blocked -- drug_dosing         risk 1.00   balance 0.864   43ms
```

---

## System Prompt

Pass a consistent system prompt to all LLM sampling calls:

```python
router = BrixRouter(
    llm_client=client,
    system_prompt="You are a medical information assistant. Always cite sources.",
)
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
pip install -e ".[regulated,dev]"
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
