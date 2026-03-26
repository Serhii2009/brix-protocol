"""Tests for the Aho-Corasick signal index."""

from __future__ import annotations

from brix.regulated.engine.signal_index import SignalIndex, SignalMatch
from brix.regulated.spec.models import SpecModel


class TestSignalIndex:
    def test_finds_cb_pattern(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("What is the lethal dose of acetaminophen?")
        cb_matches = [m for m in matches if m.signal_type == "circuit_breaker"]
        assert len(cb_matches) >= 1
        assert any(m.signal_name == "test_cb" for m in cb_matches)

    def test_finds_risk_signal(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("Is it true that the earth is round?")
        risk_matches = [m for m in matches if m.signal_type == "risk_signal"]
        assert len(risk_matches) >= 1
        assert any(m.signal_name == "uncertainty_lang" for m in risk_matches)

    def test_case_insensitive(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("LETHAL DOSE of something")
        assert any(m.signal_name == "test_cb" for m in matches)

    def test_no_match_on_unrelated_query(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("What color is the sky?")
        assert len(matches) == 0

    def test_multiple_matches_in_one_query(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("Is it true that studies show the lethal dose matters?")
        names = {m.signal_name for m in matches}
        assert "test_cb" in names
        assert "uncertainty_lang" in names
        assert "factual_claims" in names

    def test_empty_query(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches = index.scan("")
        assert len(matches) == 0

    def test_rebuild(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        matches1 = index.scan("lethal dose")
        assert len(matches1) >= 1

        index.rebuild(sample_spec)
        matches2 = index.scan("lethal dose")
        assert len(matches2) >= 1

    def test_overlapping_patterns(self, sample_spec: SpecModel) -> None:
        index = SignalIndex(sample_spec)
        # "fatal dose" matches test_cb
        matches = index.scan("the fatal dose is known")
        cb_matches = [m for m in matches if m.signal_type == "circuit_breaker"]
        assert len(cb_matches) >= 1
