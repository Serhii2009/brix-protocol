"""BRIX Quickstart — Three scenarios demonstrating the core pipeline.

Run with: python examples/quickstart.py

Demonstrates:
  1. Circuit breaker firing on a medical dosing query
  2. Epistemic uncertainty classified and signaling retrieval needed
  3. Safe general knowledge query passing through without intervention
"""

from __future__ import annotations

import asyncio
import json

from brix.analysis.consistency import ConsistencyResult
from brix.core.router import BrixRouter
from brix.llm.mock import MockLLMClient


class QuickstartAnalyzer:
    """Mock analyzer for the quickstart — skips model loading."""

    def __init__(self, mean_similarity: float = 0.95, variance: float = 0.01) -> None:
        self._mean = mean_similarity
        self._variance = variance

    def analyze(self, samples: list[str]) -> ConsistencyResult:
        n = len(samples)
        count = max(1, n * (n - 1) // 2)
        return ConsistencyResult(
            mean_similarity=self._mean,
            variance=self._variance,
            pairwise_similarities=[self._mean] * count,
        )


def print_result(scenario: str, result) -> None:
    """Pretty-print a StructuredResult."""
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {scenario}")
    print(f"{'='*70}")
    data = json.loads(result.model_dump_json())
    for key, value in data.items():
        print(f"  {key:30s} : {value}")
    print()


async def main() -> None:
    print("\n" + "=" * 70)
    print("  BRIX Quickstart — Runtime Reliability Infrastructure for LLM Pipelines")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # Scenario 1: Circuit Breaker fires on medical dosing query
    # ─────────────────────────────────────────────────────────────────────
    mock_medical = MockLLMClient(
        responses=[
            "I cannot provide specific dosage information. Please consult a medical professional.",
            "As an AI, I'm not qualified to advise on drug dosages. Seek professional medical advice.",
            "I must decline to answer questions about lethal doses. Please consult a healthcare provider.",
        ]
    )
    router1 = BrixRouter(llm_client=mock_medical, _analyzer=QuickstartAnalyzer())
    result1 = await router1.process("What is the lethal dose of acetaminophen for an adult?")
    print_result("1 — Circuit Breaker: Medical Dosing Query", result1)

    # ─────────────────────────────────────────────────────────────────────
    # Scenario 2: Epistemic uncertainty — retrieval needed
    # ─────────────────────────────────────────────────────────────────────
    mock_epistemic = MockLLMClient(
        responses=[
            "I'm not certain about the exact current tax deduction limits. Tax laws change frequently.",
            "I cannot confirm the precise deduction amounts as they may have been updated recently.",
        ]
    )
    # Use an analyzer that produces high consistency with refusals (→ EPISTEMIC)
    router2 = BrixRouter(
        llm_client=mock_epistemic,
        _analyzer=QuickstartAnalyzer(mean_similarity=0.92, variance=0.01),
    )
    result2 = await router2.process(
        "Can you confirm the exact tax deduction for home office expenses in 2026?"
    )
    print_result("2 — Epistemic Uncertainty: Tax Information Query", result2)

    # ─────────────────────────────────────────────────────────────────────
    # Scenario 3: Safe query — passthrough without intervention
    # ─────────────────────────────────────────────────────────────────────
    mock_safe = MockLLMClient(
        default_response="The sky appears blue because of Rayleigh scattering. "
        "Shorter blue wavelengths of sunlight are scattered more than longer red "
        "wavelengths by the nitrogen and oxygen molecules in Earth's atmosphere, "
        "making the sky appear blue to observers on the ground."
    )
    router3 = BrixRouter(llm_client=mock_safe, _analyzer=QuickstartAnalyzer())
    result3 = await router3.process("Why is the sky blue?")
    print_result("3 — Safe Passthrough: General Knowledge Query", result3)

    # Summary
    print("=" * 70)
    print("  QUICKSTART COMPLETE")
    print(
        f"  Scenario 1: Circuit breaker {'FIRED' if result1.circuit_breaker_hit else 'did not fire'}"
    )
    print(
        f"  Scenario 2: Uncertainty type = {result2.uncertainty_type}, action = {result2.action_taken}"
    )
    print(
        f"  Scenario 3: Passed through = {not result3.intervention_necessary}, action = {result3.action_taken}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
