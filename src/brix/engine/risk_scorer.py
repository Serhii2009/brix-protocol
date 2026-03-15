"""Risk Score Track — weighted aggregation of risk signals.

Computes the aggregate risk score using the formula:
  risk_score = max(registered_signals) * 1.0
             + sum(universal_signals) * 0.6
             + retrieval_score_penalty * 0.8

Where retrieval_score_penalty = max(0, 0.85 - retrieval_score)
if retrieval_score is provided, else 0.

This track has NO shared mutable state with the Circuit Breaker Track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brix.engine.signal_index import SignalIndex
from brix.spec.models import SpecModel


@dataclass(frozen=True, slots=True)
class RiskScoreResult:
    """Result of risk score evaluation."""

    score: float
    signals_triggered: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


class RiskScoreTrack:
    """Weighted risk score evaluation — graduated, not binary.

    Completely independent from the Circuit Breaker Track. Uses the shared
    SignalIndex (read-only) for pattern matching but maintains no mutable
    state that could be shared.
    """

    def __init__(self, spec: SpecModel, signal_index: SignalIndex) -> None:
        self._signal_index = signal_index
        # Build lookups for signal metadata
        self._weight_map: dict[str, float] = {}
        self._category_map: dict[str, str] = {}
        self._exclude_map: dict[str, list[str]] = {}
        for signal in spec.risk_signals:
            self._weight_map[signal.name] = signal.weight
            self._category_map[signal.name] = signal.category
            self._exclude_map[signal.name] = [p.lower() for p in signal.exclude_context]

    def evaluate(
        self,
        query: str,
        context: str | None = None,
        retrieval_score: float | None = None,
    ) -> RiskScoreResult:
        """Compute the aggregate risk score for a query.

        Args:
            query: The user query text.
            context: Optional context string for exclude_context filtering.
            retrieval_score: Optional RAG retrieval quality score (0.0–1.0).

        Returns:
            RiskScoreResult with the computed score and breakdown.
        """
        matches = self._signal_index.scan(query)
        risk_matches = [m for m in matches if m.signal_type == "risk_signal"]

        # Apply exclude_context post-match filtering
        query_lower = query.lower()
        context_lower = context.lower() if context else ""
        combined_text = f"{query_lower} {context_lower}"

        surviving_names: set[str] = set()
        for match in risk_matches:
            exclusions = self._exclude_map.get(match.signal_name, [])
            cancelled = any(exc in combined_text for exc in exclusions)
            if not cancelled:
                surviving_names.add(match.signal_name)

        # Separate registered vs universal signals
        registered_weights: list[float] = []
        universal_weights: list[float] = []
        for name in surviving_names:
            weight = self._weight_map.get(name, 0.0)
            category = self._category_map.get(name, "registered")
            if category == "universal":
                universal_weights.append(weight)
            else:
                registered_weights.append(weight)

        # Compute risk score components
        max_registered = max(registered_weights) if registered_weights else 0.0
        sum_universal = sum(universal_weights)
        retrieval_penalty = max(0.0, 0.85 - retrieval_score) if retrieval_score is not None else 0.0

        registered_component = max_registered * 1.0
        universal_component = sum_universal * 0.6
        retrieval_component = retrieval_penalty * 0.8

        raw_score = registered_component + universal_component + retrieval_component
        clamped_score = min(max(raw_score, 0.0), 1.0)

        return RiskScoreResult(
            score=clamped_score,
            signals_triggered=sorted(surviving_names),
            breakdown={
                "max_registered": max_registered,
                "registered_component": registered_component,
                "sum_universal": sum_universal,
                "universal_component": universal_component,
                "retrieval_penalty": retrieval_penalty,
                "retrieval_component": retrieval_component,
                "raw_score": raw_score,
                "clamped_score": clamped_score,
            },
        )
