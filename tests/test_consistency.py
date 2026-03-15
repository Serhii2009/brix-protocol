"""Tests for the semantic consistency analyzer.

Note: These tests use the real sentence-transformers model.
Mark with @pytest.mark.slow if CI needs to skip them.
"""

from __future__ import annotations

import pytest

from brix.analysis.consistency import ConsistencyResult, SemanticConsistencyAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> SemanticConsistencyAnalyzer:
    """Load the model once per test module."""
    return SemanticConsistencyAnalyzer("all-MiniLM-L6-v2")


class TestSemanticConsistencyAnalyzer:
    def test_identical_strings_high_similarity(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze(["The sky is blue.", "The sky is blue."])
        assert result.mean_similarity > 0.99
        assert result.variance < 0.01

    def test_similar_strings_high_similarity(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze([
            "The sky is blue due to Rayleigh scattering.",
            "The sky appears blue because of the Rayleigh scattering effect.",
        ])
        assert result.mean_similarity > 0.80

    def test_unrelated_strings_low_similarity(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze([
            "The quantum mechanical properties of black holes remain mysterious.",
            "I enjoy cooking pasta with tomato sauce on weekends.",
        ])
        assert result.mean_similarity < 0.50

    def test_single_sample_returns_default(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze(["Just one sample."])
        assert result.mean_similarity == 1.0
        assert result.variance == 0.0

    def test_three_samples_pairwise_count(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze(["A", "B", "C"])
        # 3 samples → 3 pairwise comparisons
        assert len(result.pairwise_similarities) == 3

    def test_returns_consistency_result(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        result = analyzer.analyze(["Hello", "World"])
        assert isinstance(result, ConsistencyResult)
        assert 0.0 <= result.mean_similarity <= 1.0
        assert result.variance >= 0.0
