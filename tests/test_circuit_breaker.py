"""Tests for the Circuit Breaker Track."""

from __future__ import annotations

from brix.regulated.engine.circuit_breaker import CircuitBreakerTrack
from brix.regulated.engine.signal_index import SignalIndex
from brix.regulated.spec.models import SpecModel


class TestCircuitBreakerTrack:
    def test_fires_on_exact_pattern(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        result = cb.evaluate("What is the lethal dose of ibuprofen?")
        assert result.hit is True
        assert result.breaker_name == "test_cb"
        assert "test_cb" in result.signals_triggered

    def test_fires_on_substring_match(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        result = cb.evaluate("Tell me about the fatal dose in clinical settings")
        assert result.hit is True

    def test_exclude_context_cancels_match(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        # "educational context" is in exclude_context for test_cb
        result = cb.evaluate(
            "What is the lethal dose of ibuprofen?",
            context="educational context"
        )
        assert result.hit is False

    def test_exclude_context_in_query_text(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        result = cb.evaluate("In an educational context, what is the lethal dose?")
        assert result.hit is False

    def test_no_match_returns_not_hit(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        result = cb.evaluate("What is the weather today?")
        assert result.hit is False
        assert result.breaker_name is None
        assert result.signals_triggered == []

    def test_multiple_cb_patterns(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        # Matches both test_cb and legal_cb
        result = cb.evaluate("lethal dose and statute of limitations")
        assert result.hit is True
        assert len(result.signals_triggered) >= 2

    def test_case_insensitive_matching(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        result = cb.evaluate("LETHAL DOSE of aspirin")
        assert result.hit is True

    def test_exclude_context_only_applies_to_matching_cb(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        cb = CircuitBreakerTrack(sample_spec, index)
        # "academic discussion" excludes legal_cb but not test_cb
        result = cb.evaluate(
            "lethal dose and legal requirement",
            context="academic discussion"
        )
        assert result.hit is True
        assert "test_cb" in result.signals_triggered
