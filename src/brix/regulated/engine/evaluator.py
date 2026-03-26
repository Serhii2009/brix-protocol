"""Two-track evaluator orchestrating Circuit Breaker and Risk Score tracks.

The Circuit Breaker Track is always evaluated first. If it fires, the
Risk Score Track is never called. The two tracks share no mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brix.regulated.engine.circuit_breaker import CircuitBreakerResult, CircuitBreakerTrack
from brix.regulated.engine.risk_scorer import RiskScoreResult, RiskScoreTrack
from brix.regulated.engine.signal_index import SignalIndex
from brix.regulated.spec.models import SpecModel


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Combined result of the two-track evaluation."""

    circuit_breaker_hit: bool
    circuit_breaker_name: str | None = None
    risk_score: float = 0.0
    signals_triggered: list[str] = field(default_factory=list)
    risk_breakdown: dict[str, float] = field(default_factory=dict)


class TwoTrackEvaluator:
    """Orchestrates the Circuit Breaker Track and Risk Score Track.

    The two tracks are architecturally separate. The evaluator runs
    the CB track first; if it fires, the risk track is skipped entirely.
    """

    def __init__(self, spec: SpecModel, signal_index: SignalIndex) -> None:
        self._cb_track = CircuitBreakerTrack(spec, signal_index)
        self._risk_track = RiskScoreTrack(spec, signal_index)

    def evaluate(
        self,
        query: str,
        context: str | None = None,
        retrieval_score: float | None = None,
    ) -> EvaluationResult:
        """Evaluate a query through both tracks sequentially.

        Args:
            query: The user query text.
            context: Optional context for exclude_context filtering.
            retrieval_score: Optional RAG retrieval quality score.

        Returns:
            EvaluationResult with circuit breaker and risk score data.
        """
        # Track 1: Circuit Breaker (always first)
        cb_result: CircuitBreakerResult = self._cb_track.evaluate(query, context)

        if cb_result.hit:
            # CB fired — do NOT evaluate risk track
            return EvaluationResult(
                circuit_breaker_hit=True,
                circuit_breaker_name=cb_result.breaker_name,
                risk_score=1.0,  # Maximum risk when CB fires
                signals_triggered=cb_result.signals_triggered,
            )

        # Track 2: Risk Score (only if CB did not fire)
        risk_result: RiskScoreResult = self._risk_track.evaluate(
            query, context, retrieval_score
        )

        return EvaluationResult(
            circuit_breaker_hit=False,
            risk_score=risk_result.score,
            signals_triggered=risk_result.signals_triggered,
            risk_breakdown=risk_result.breakdown,
        )
