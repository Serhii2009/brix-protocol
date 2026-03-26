"""BrixRouter — the main entry point for the BRIX runtime pipeline.

Orchestrates the complete request processing flow: two-track evaluation,
adaptive sampling, semantic consistency analysis, uncertainty classification,
action execution, and structured result assembly.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from brix.regulated.actions.executor import ActionExecutor
from brix.regulated.console.output import print_result
from brix.regulated.analysis.classifier import UncertaintyClassifier
from brix.regulated.analysis.consistency import SemanticConsistencyAnalyzer
from brix.regulated.balance.tracker import BalanceTracker
from brix.regulated.core.result import ActionTaken, StructuredResult, UncertaintyType
from brix.regulated.engine.evaluator import TwoTrackEvaluator
from brix.regulated.engine.signal_index import SignalIndex
from brix.regulated.llm.protocol import LLMClient
from brix.regulated.output.guard import OutputGuard
from brix.regulated.output.result import OutputResult
from brix.regulated.retrieval.protocol import RetrievalProvider
from brix.regulated.sampling.sampler import AdaptiveSampler
from brix.regulated.spec.defaults import get_default_spec_path
from brix.regulated.spec.loader import load_spec
from brix.regulated.spec.models import SpecModel


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
        system_prompt: str | None = None,
        enable_output_guard: bool = False,
        retrieval_provider: RetrievalProvider | None = None,
        _analyzer: SemanticConsistencyAnalyzer | None = None,
    ) -> None:
        """Initialize the BRIX router.

        Args:
            llm_client: Any LLM client implementing the LLMClient protocol.
            spec: A SpecModel, path to YAML file, or None for built-in default.
            embedding_model: Sentence-transformers model name (loaded once).
            log_path: Optional path for JSONL structured result logging.
            system_prompt: Optional system prompt passed to all LLM sample calls.
            enable_output_guard: Enable response-side output scanning.
            retrieval_provider: Optional RAG provider for real retrieval execution.
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
        self._system_prompt = system_prompt

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
        self._executor = ActionExecutor(
            self._spec, self._llm, retrieval_provider=retrieval_provider
        )
        self._balance = BalanceTracker(
            risk_threshold=self._spec.sampling_config.low_threshold,
        )

        # Output guard
        self._output_guard: OutputGuard | None = None
        if enable_output_guard:
            self._output_guard = OutputGuard(
                self._spec, _analyzer=self._analyzer
            )

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
            retrieval_score: Optional RAG retrieval quality score (0.0-1.0).

        Returns:
            A complete StructuredResult with all fields populated.

        Raises:
            ValueError: If retrieval_score is not None and outside [0.0, 1.0].
        """
        if retrieval_score is not None and not (0.0 <= retrieval_score <= 1.0):
            raise ValueError(
                f"retrieval_score must be between 0.0 and 1.0, got {retrieval_score}"
            )

        t0 = time.perf_counter()
        decision_id = uuid4()

        # Step 1: Two-track evaluation
        eval_result = self._evaluator.evaluate(query, context, retrieval_score)

        # Step 2: Adaptive sampling
        sampler_result = await self._sampler.collect(
            query=query,
            risk_score=eval_result.risk_score,
            circuit_breaker_hit=eval_result.circuit_breaker_hit,
            system=self._system_prompt,
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

        # Step 5: Output guard (if enabled)
        output_result: OutputResult | None = None
        if self._output_guard is not None:
            output_result = await self._output_guard.analyze(
                action_result.response, query=query, context=context
            )
            # If output blocked and no input-side intervention, escalate
            if output_result.output_blocked and not action_result.intervention_necessary:
                action_result = dataclasses.replace(
                    action_result,
                    response_requires_verification=True,
                    intervention_necessary=True,
                )

        # Step 6: Balance Index update
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
            response_requires_verification=action_result.response_requires_verification,
            unverified_draft=action_result.unverified_draft,
            sampler_partial_failure=sampler_result.partial_failure,
            retrieval_executed=action_result.retrieval_executed,
            retrieval_failed=action_result.retrieval_failed,
            retrieval_sources=action_result.retrieval_sources,
            output_result=output_result,
        )

        # Optional JSONL logging
        if self._log_path is not None:
            self._write_log(result)

        # Console feedback (never propagates errors)
        try:
            print_result(result, output_result=output_result)
        except Exception:
            pass

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
        try:
            line = result.model_dump_json() + "\n"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            print(f"Warning: failed to write BRIX log: {exc}", file=sys.stderr)
