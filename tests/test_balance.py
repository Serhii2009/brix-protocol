# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for the Balance Index tracker — harmonic mean and counter correctness."""

from __future__ import annotations

from uuid import uuid4

import pytest

from brix.regulated.balance.tracker import BalanceTracker


class TestBalanceTracker:
    def test_initial_balance_zero(self) -> None:
        tracker = BalanceTracker()
        assert tracker.compute_balance_index() == 0.0

    def test_perfect_balance(self) -> None:
        tracker = BalanceTracker()
        # 10 TPs and 10 TNs → R=1.0, U=1.0 → Balance=1.0
        for _ in range(10):
            tracker.record_decision(uuid4(), True, True, 0.8)  # TP
        for _ in range(10):
            tracker.record_decision(uuid4(), False, False, 0.1)  # TN
        assert tracker.compute_balance_index() == pytest.approx(1.0)

    def test_half_balance(self) -> None:
        tracker = BalanceTracker()
        # 5 TP + 5 FN → R=0.5; 5 TN + 5 FP → U=0.5
        for _ in range(5):
            tracker.record_decision(uuid4(), True, True, 0.8)  # TP
        for _ in range(5):
            tracker.record_decision(uuid4(), False, True, 0.8)  # FN (risky, not intervened)
        for _ in range(5):
            tracker.record_decision(uuid4(), False, False, 0.1)  # TN
        for _ in range(5):
            tracker.record_decision(uuid4(), True, False, 0.1)  # FP (safe, intervened)
        balance = tracker.compute_balance_index()
        # harmonic mean of 0.5 and 0.5 = 0.5
        assert balance == pytest.approx(0.5)

    def test_zero_reliability(self) -> None:
        tracker = BalanceTracker()
        # All risky queries missed → R=0
        for _ in range(10):
            tracker.record_decision(uuid4(), False, True, 0.8)  # FN
        for _ in range(10):
            tracker.record_decision(uuid4(), False, False, 0.1)  # TN
        assert tracker.compute_balance_index() == pytest.approx(0.0)

    def test_zero_utility(self) -> None:
        tracker = BalanceTracker()
        # All safe queries blocked → U=0
        for _ in range(10):
            tracker.record_decision(uuid4(), True, True, 0.8)  # TP
        for _ in range(10):
            tracker.record_decision(uuid4(), True, False, 0.1)  # FP
        assert tracker.compute_balance_index() == pytest.approx(0.0)

    def test_feedback_corrects_heuristic(self) -> None:
        tracker = BalanceTracker()
        decision_id = uuid4()
        # Heuristic: low risk + no intervention → TN
        tracker.record_decision(decision_id, False, False, 0.1)
        assert tracker.state.tn == 1

        # Feedback: it was actually risky → should become FN
        tracker.feedback(decision_id, was_intervention_necessary=True)
        assert tracker.state.tn == 0
        assert tracker.state.fn == 1

    def test_feedback_unknown_id_ignored(self) -> None:
        tracker = BalanceTracker()
        tracker.record_decision(uuid4(), False, False, 0.1)
        # Feedback for unknown ID → no crash, no change
        tracker.feedback(uuid4(), was_intervention_necessary=True)
        assert tracker.state.tn == 1

    def test_counter_values(self) -> None:
        tracker = BalanceTracker()
        id1 = uuid4()
        tracker.record_decision(id1, True, True, 0.8)  # TP
        assert tracker.state.tp == 1
        assert tracker.state.fn == 0
        assert tracker.state.tn == 0
        assert tracker.state.fp == 0

    def test_harmonic_mean_formula(self) -> None:
        tracker = BalanceTracker()
        # R=0.8, U=0.6 → 2*0.8*0.6/(0.8+0.6) = 0.96/1.4 ≈ 0.6857
        # 8 TP + 2 FN → R=0.8; 6 TN + 4 FP → U=0.6
        for _ in range(8):
            tracker.record_decision(uuid4(), True, True, 0.8)
        for _ in range(2):
            tracker.record_decision(uuid4(), False, True, 0.8)
        for _ in range(6):
            tracker.record_decision(uuid4(), False, False, 0.1)
        for _ in range(4):
            tracker.record_decision(uuid4(), True, False, 0.1)
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert tracker.compute_balance_index() == pytest.approx(expected, abs=0.001)
