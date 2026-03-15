"""BrixRouter — the main entry point for the BRIX runtime pipeline.

Orchestrates the complete request processing flow: two-track evaluation,
adaptive sampling, semantic consistency analysis, uncertainty classification,
action execution, and structured result assembly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from brix.actions.executor import ActionExecutor
from brix.analysis.classifier import UncertaintyClassifier
from brix.analysis.consistency import SemanticConsistencyAnalyzer
from brix.balance.tracker import BalanceTracker
from brix.core.result import ActionTaken, StructuredResult, UncertaintyType
from brix.engine.evaluator import TwoTrackEvaluator
from brix.engine.signal_index import SignalIndex
from brix.llm.protocol import LLMClient
from brix.sampling.sampler import AdaptiveSampler
from brix.spec.defaults import get_default_spec_path
from brix.spec.loader import load_spec
from brix.spec.models import SpecModel


class BrixRouter:
    """Main BRIX runtime router — wraps any LLM client with reliability infrastructure.

    Processes every query through the two-track evaluation system, adaptive
    sampling, semantic consistency analysis, and action execution. Returns
    a StructuredResult for every request.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        spec: SpecModel | str | Path | None = None,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        log_path: Path | None = None,
        _analyzer: SemanticConsistencyAnalyzer | None = None,
    ) -> None:
        """Initialize the BRIX router.

        Args:
            llm_client: Any LLM client implementing the LLMClient protocol.
            spec: A SpecModel, path to YAML file, or None for built-in default.
            embedding_model: Sentence-transformers model name (loaded once).
            log_path: Optional path for JSONL structured result logging.
            _analyzer: Internal override for testing (skip model loading).
        """
        # Load specification
        if spec is None:
            self._spec = load_spec(get_default_spec_path())
        elif isinstance(spec, SpecModel):
            self._spec = spec
        else:
            self._spec = load_spec(spec)

        self._llm = llm_client
        self._log_path = log_path

        # Build components
        self._signal_index = SignalIndex(self._spec)
        self._evaluator = TwoTrackEvaluator(self._spec, self._signal_index)
        self._sampler = AdaptiveSampler(self._llm, self._spec.sampling_config)

        # Analyzer: use injected mock for testing or load real model
        if _analyzer is not None:
            self._analyzer = _analyzer
        else:
            self._analyzer = SemanticConsistencyAnalyzer(embedding_model)

        self._classifier = UncertaintyClassifier(self._analyzer)
        self._executor = ActionExecutor(self._spec, self._llm)
        self._balance = BalanceTracker()

        # Resolve model compatibility status
        self._registry_version = f"{self._spec.metadata.name}/{self._spec.metadata.version}"
        self._compat_status = self._resolve_compat_status()

    async def process(
        self,
        query: str,
        *,
        context: str | None = None,
        retrieval_score: float | None = None,
    ) -> StructuredResult:
        """Process a single query through the full BRIX pipeline.

        Args:
            query: The user query text.
            context: Optional context for exclude_context filtering.
            retrieval_score: Optional RAG retrieval quality score (0.0–1.0).

        Returns:
            A complete StructuredResult with all fields populated.
        """
        t0 = time.perf_counter()
        decision_id = uuid4()

        # Step 1: Two-track evaluation
        eval_result = self._evaluator.evaluate(query, context, retrieval_score)

        # Step 2: Adaptive sampling
        sampler_result = await self._sampler.collect(
            query=query,
            risk_score=eval_result.risk_score,
            circuit_breaker_hit=eval_result.circuit_breaker_hit,
        )

        # Step 3: Uncertainty classification
        classification = self._classifier.classify(sampler_result.samples)

        # Override: if CB hit, ensure we treat appropriately
        uncertainty_type = classification.uncertainty_type
        if eval_result.circuit_breaker_hit and uncertainty_type == UncertaintyType.CERTAIN:
            uncertainty_type = UncertaintyType.EPISTEMIC

        # Step 4: Action execution
        action_result = await self._executor.execute(
            uncertainty_type=uncertainty_type,
            samples=sampler_result.samples,
            query=query,
            force_retrieval=sampler_result.force_retrieval,
        )

        # Step 5: Balance Index update
        reliability_signal, utility_signal, balance_index = self._balance.record_decision(
            decision_id=decision_id,
            intervention_necessary=action_result.intervention_necessary,
            circuit_breaker_hit=eval_result.circuit_breaker_hit,
            risk_score=eval_result.risk_score,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = StructuredResult(
            decision_id=decision_id,
            uncertainty_type=uncertainty_type,
            subtype=classification.subtype,
            action_taken=action_result.action_taken,
            response=action_result.response,
            circuit_breaker_hit=eval_result.circuit_breaker_hit,
            circuit_breaker_name=eval_result.circuit_breaker_name,
            signals_triggered=eval_result.signals_triggered,
            risk_score=eval_result.risk_score,
            reliability_signal=reliability_signal,
            utility_signal=utility_signal,
            balance_index=balance_index,
            intervention_necessary=action_result.intervention_necessary,
            registry_version=self._registry_version,
            model_compatibility_status=self._compat_status,
            cost_tokens_extra=action_result.cost_tokens_extra,
            latency_ms=latency_ms,
        )

        # Optional JSONL logging
        if self._log_path is not None:
            self._write_log(result)

        return result

    def feedback(self, decision_id: UUID, was_intervention_necessary: bool) -> None:
        """Provide ground-truth feedback for a previous decision.

        Updates the Balance Index tracker with the true label,
        correcting the heuristic classification.

        Args:
            decision_id: UUID of the decision to correct.
            was_intervention_necessary: True if intervention was actually needed.
        """
        self._balance.feedback(decision_id, was_intervention_necessary)

    @property
    def balance_index(self) -> float:
        """Current running Balance Index for this session."""
        return self._balance.compute_balance_index()

    @property
    def balance_state(self) -> dict[str, int]:
        """Current TP/FN/TN/FP counters."""
        state = self._balance.state
        return {"tp": state.tp, "fn": state.fn, "tn": state.tn, "fp": state.fp}

    def _resolve_compat_status(self) -> Literal["verified", "community", "untested", "unknown"]:
        """Determine model compatibility status from spec metadata."""
        if not self._spec.metadata.model_compatibility:
            return "unknown"
        # Return the highest status found
        statuses = {mc.status for mc in self._spec.metadata.model_compatibility}
        if "verified" in statuses:
            return "verified"
        if "community" in statuses:
            return "community"
        if "untested" in statuses:
            return "untested"
        return "unknown"

    def _write_log(self, result: StructuredResult) -> None:
        """Append a StructuredResult as a JSONL line to the log file."""
        if self._log_path is None:
            return
        line = result.model_dump_json() + "\n"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)
