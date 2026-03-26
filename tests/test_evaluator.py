"""Tests for the two-track evaluator."""

from __future__ import annotations

from brix.regulated.engine.evaluator import TwoTrackEvaluator
from brix.regulated.engine.signal_index import SignalIndex
from brix.regulated.spec.models import SpecModel


class TestTwoTrackEvaluator:
    def test_cb_fires_skips_risk(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        evaluator = TwoTrackEvaluator(sample_spec, index)
        result = evaluator.evaluate("What is the lethal dose of aspirin?")
        assert result.circuit_breaker_hit is True
        assert result.risk_score == 1.0  # Max when CB fires
        assert "test_cb" in result.signals_triggered
        # risk_breakdown should be empty since risk track was skipped
        assert result.risk_breakdown == {}

    def test_no_cb_evaluates_risk(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        evaluator = TwoTrackEvaluator(sample_spec, index)
        result = evaluator.evaluate("is it true that something happened?")
        assert result.circuit_breaker_hit is False
        assert result.risk_score > 0
        assert len(result.risk_breakdown) > 0

    def test_clean_query_no_signals(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        evaluator = TwoTrackEvaluator(sample_spec, index)
        result = evaluator.evaluate("What color is the sky?")
        assert result.circuit_breaker_hit is False
        assert result.risk_score == 0.0
        assert result.signals_triggered == []

    def test_cb_with_exclude_context_falls_to_risk(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        evaluator = TwoTrackEvaluator(sample_spec, index)
        # CB excluded, but risk signals may still fire
        result = evaluator.evaluate(
            "In an educational context, is it true that the lethal dose matters?"
        )
        # CB should not fire due to exclude_context
        assert result.circuit_breaker_hit is False
        # But risk signals should still be evaluated
        assert result.risk_score >= 0

    def test_retrieval_score_passed_to_risk(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        evaluator = TwoTrackEvaluator(sample_spec, index)
        result = evaluator.evaluate("What color is the sky?", retrieval_score=0.50)
        # Low retrieval score adds penalty
        assert result.risk_score > 0
        assert result.risk_breakdown["retrieval_penalty"] > 0
