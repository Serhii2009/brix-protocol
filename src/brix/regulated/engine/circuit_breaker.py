"""Circuit Breaker Track — binary, deterministic enforcement.

When a query matches any circuit breaker pattern and no exclude_context
term cancels the match, the circuit breaker fires unconditionally.
No gradation, no weighting. This track has NO shared mutable state
with the Risk Score Track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brix.regulated.engine.signal_index import SignalIndex, SignalMatch, _normalize
from brix.regulated.spec.models import SpecModel


@dataclass(frozen=True, slots=True)
class CircuitBreakerResult:
    """Result of circuit breaker evaluation."""

    hit: bool
    breaker_name: str | None = None
    signals_triggered: list[str] = field(default_factory=list)


class CircuitBreakerTrack:
    """Binary circuit breaker evaluation — fires or does not fire.

    Completely independent from the Risk Score Track. Uses the shared
    SignalIndex (read-only) for pattern matching but maintains no
    mutable state that could be shared.
    """

    def __init__(self, spec: SpecModel, signal_index: SignalIndex) -> None:
        self._signal_index = signal_index
        # Build a lookup of CB name -> exclude_context patterns
        self._exclude_map: dict[str, list[str]] = {}
        for cb in spec.circuit_breakers:
            self._exclude_map[cb.name] = [_normalize(p).lower() for p in cb.exclude_context]

    def evaluate(self, query: str, context: str | None = None) -> CircuitBreakerResult:
        """Evaluate a query against all circuit breaker patterns.

        Args:
            query: The user query text.
            context: Optional context string for exclude_context filtering.

        Returns:
            CircuitBreakerResult indicating whether any breaker fired.
        """
        matches = self._signal_index.scan(query)
        cb_matches = [m for m in matches if m.signal_type == "circuit_breaker"]

        if not cb_matches:
            return CircuitBreakerResult(hit=False)

        # Apply exclude_context: if ANY exclusion term is present in the
        # query text, that CB match is cancelled.
        combined_text = _normalize(f"{query} {context or ''}").lower()

        surviving: list[SignalMatch] = []
        for match in cb_matches:
            exclusions = self._exclude_map.get(match.signal_name, [])
            cancelled = any(exc in combined_text for exc in exclusions)
            if not cancelled:
                surviving.append(match)

        if not surviving:
            return CircuitBreakerResult(hit=False)

        # Deduplicate signal names
        triggered_names = list(dict.fromkeys(m.signal_name for m in surviving))
        return CircuitBreakerResult(
            hit=True,
            breaker_name=triggered_names[0],
            signals_triggered=triggered_names,
        )
