"""Balance Index tracker — TP/FN/TN/FP counters and harmonic mean computation.

The Balance Index is defined as:
  Balance Index = 2 * (R * U) / (R + U)
  where R = TP / (TP + FN), U = TN / (TN + FP)

Supports both heuristic defaults and explicit feedback from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class BalanceState:
    """Current state of the balance tracker counters."""

    tp: int = 0  # Risky query correctly intercepted
    fn: int = 0  # Risky query passed without intervention
    tn: int = 0  # Safe query correctly passed
    fp: int = 0  # Safe query incorrectly intercepted


class BalanceTracker:
    """Tracks reliability and utility metrics across a session.

    Computes the running Balance Index after every request. Supports
    both heuristic auto-classification and explicit feedback via
    the feedback() method.
    """

    def __init__(self) -> None:
        self._state = BalanceState()
        self._pending: dict[UUID, _PendingDecision] = {}

    @property
    def state(self) -> BalanceState:
        """Current counter state."""
        return self._state

    def record_decision(
        self,
        decision_id: UUID,
        intervention_necessary: bool,
        circuit_breaker_hit: bool,
        risk_score: float,
    ) -> tuple[bool, bool, float]:
        """Record a decision using heuristic classification.

        Heuristic rules:
        - CB hit + intervention → TP (assumed risky, correctly caught)
        - High risk (>0.70) + intervention → TP
        - Medium risk + intervention → TP (conservative)
        - Low risk + no intervention → TN (safe, correctly passed)
        - Low risk + intervention → FP (safe, incorrectly caught)
        - High risk + no intervention → FN (risky, missed)

        Also stores the decision for possible later feedback override.

        Args:
            decision_id: Unique ID for this decision.
            intervention_necessary: Whether intervention was applied.
            circuit_breaker_hit: Whether a CB fired.
            risk_score: Computed risk score.

        Returns:
            Tuple of (reliability_signal, utility_signal, balance_index).
        """
        is_risky = circuit_breaker_hit or risk_score > 0.40

        if is_risky and intervention_necessary:
            self._state.tp += 1
            reliability_signal = True
            utility_signal = True
        elif is_risky and not intervention_necessary:
            self._state.fn += 1
            reliability_signal = False
            utility_signal = True
        elif not is_risky and not intervention_necessary:
            self._state.tn += 1
            reliability_signal = True
            utility_signal = True
        else:  # not risky, but intervention happened
            self._state.fp += 1
            reliability_signal = True
            utility_signal = False

        # Store for potential feedback override
        self._pending[decision_id] = _PendingDecision(
            is_risky=is_risky,
            intervention_applied=intervention_necessary,
        )

        balance = self.compute_balance_index()
        return reliability_signal, utility_signal, balance

    def feedback(self, decision_id: UUID, was_intervention_necessary: bool) -> None:
        """Provide ground-truth feedback to correct heuristic classification.

        Args:
            decision_id: The decision to correct.
            was_intervention_necessary: True if intervention was actually needed.
        """
        pending = self._pending.pop(decision_id, None)
        if pending is None:
            return

        # Reverse the heuristic classification
        if pending.is_risky and pending.intervention_applied:
            self._state.tp -= 1
        elif pending.is_risky and not pending.intervention_applied:
            self._state.fn -= 1
        elif not pending.is_risky and not pending.intervention_applied:
            self._state.tn -= 1
        else:
            self._state.fp -= 1

        # Apply ground-truth classification
        actually_risky = was_intervention_necessary
        if actually_risky and pending.intervention_applied:
            self._state.tp += 1
        elif actually_risky and not pending.intervention_applied:
            self._state.fn += 1
        elif not actually_risky and not pending.intervention_applied:
            self._state.tn += 1
        else:
            self._state.fp += 1

    def compute_balance_index(self) -> float:
        """Compute the current Balance Index.

        Balance Index = 2 * R * U / (R + U)
        where R = TP / (TP + FN), U = TN / (TN + FP)

        Returns 0.0 if insufficient data to compute.
        """
        r = self._reliability_score()
        u = self._utility_score()
        if r + u == 0:
            return 0.0
        return 2.0 * r * u / (r + u)

    def _reliability_score(self) -> float:
        """R = TP / (TP + FN). Returns 0.0 if no relevant data."""
        total = self._state.tp + self._state.fn
        if total == 0:
            return 0.0
        return self._state.tp / total

    def _utility_score(self) -> float:
        """U = TN / (TN + FP). Returns 0.0 if no relevant data."""
        total = self._state.tn + self._state.fp
        if total == 0:
            return 0.0
        return self._state.tn / total


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    """Internal record of a decision awaiting potential feedback."""

    is_risky: bool
    intervention_applied: bool
