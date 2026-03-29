# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for the Risk Score Track — formula correctness."""

from __future__ import annotations

import pytest

from brix.regulated.engine.risk_scorer import RiskScoreTrack
from brix.regulated.engine.signal_index import SignalIndex
from brix.regulated.spec.models import SpecModel


class TestRiskScoreTrack:
    def test_no_signals_score_zero(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        result = risk.evaluate("What color is the sky?")
        assert result.score == 0.0
        assert result.signals_triggered == []

    def test_registered_signal_max_weight(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "is it true that" triggers uncertainty_lang (weight 0.6, registered)
        result = risk.evaluate("is it true that the earth is round?")
        assert "uncertainty_lang" in result.signals_triggered
        # max(registered) * 1.0 = 0.6
        assert result.breakdown["max_registered"] == pytest.approx(0.6)
        assert result.breakdown["registered_component"] == pytest.approx(0.6)

    def test_universal_signal_weight(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "exactly" triggers specific_numbers (weight 0.5, universal)
        result = risk.evaluate("The value is exactly 42")
        assert "specific_numbers" in result.signals_triggered
        # sum(universal) * 0.6 = 0.5 * 0.6 = 0.3
        assert result.breakdown["universal_component"] == pytest.approx(0.3)

    def test_retrieval_score_penalty(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # retrieval_score = 0.50 → penalty = max(0, 0.85 - 0.50) = 0.35
        # retrieval_component = 0.35 * 0.8 = 0.28
        result = risk.evaluate("What color is the sky?", retrieval_score=0.50)
        assert result.breakdown["retrieval_penalty"] == pytest.approx(0.35)
        assert result.breakdown["retrieval_component"] == pytest.approx(0.28)

    def test_high_retrieval_score_no_penalty(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        result = risk.evaluate("What color is the sky?", retrieval_score=0.90)
        assert result.breakdown["retrieval_penalty"] == pytest.approx(0.0)
        assert result.breakdown["retrieval_component"] == pytest.approx(0.0)

    def test_combined_formula(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "is it true that" → registered 0.6, "exactly" → universal 0.5
        result = risk.evaluate("is it true that it weighs exactly 10kg?", retrieval_score=0.50)
        # max_registered * 1.0 + sum_universal * 0.6 + penalty * 0.8
        # 0.6 * 1.0 + 0.5 * 0.6 + 0.35 * 0.8 = 0.6 + 0.3 + 0.28 = 1.18 → clamped to 1.0
        assert result.score == pytest.approx(1.0)

    def test_score_clamped_to_one(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # Many signals + low retrieval → should clamp
        result = risk.evaluate(
            "is it true that studies show exactly the deadline is tomorrow",
            retrieval_score=0.10,
        )
        assert result.score <= 1.0

    def test_exclude_context_filters_risk_signal(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "opinion" excludes factual_claims
        result = risk.evaluate("In my opinion, studies show interesting things")
        assert "factual_claims" not in result.signals_triggered

    def test_exclude_context_in_context_param(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "approximate" excludes specific_numbers
        result = risk.evaluate("The value is exactly 42", context="approximate value")
        assert "specific_numbers" not in result.signals_triggered

    def test_no_retrieval_score_no_penalty(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        result = risk.evaluate("is it true that something happened?")
        assert result.breakdown["retrieval_penalty"] == 0.0
        assert result.breakdown["retrieval_component"] == 0.0

    def test_multiple_registered_takes_max(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        risk = RiskScoreTrack(sample_spec, index)
        # "is it true that" (0.6) and "studies show" (0.7) both registered
        result = risk.evaluate("is it true that studies show results?")
        assert result.breakdown["max_registered"] == pytest.approx(0.7)
