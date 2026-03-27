"""Tests for the uncertainty classifier — all threshold boundary conditions."""

from __future__ import annotations

from brix.regulated.analysis.classifier import UncertaintyClassifier
from brix.regulated.analysis.consistency import ConsistencyResult
from brix.regulated.core.result import UncertaintyType


class MockAnalyzerForClassifier:
    """Mock analyzer returning preset consistency values."""

    def __init__(self, mean_similarity: float, variance: float) -> None:
        self._mean = mean_similarity
        self._variance = variance

    def analyze(self, samples: list[str]) -> ConsistencyResult:
        n = len(samples)
        count = max(1, n * (n - 1) // 2)
        return ConsistencyResult(
            mean_similarity=self._mean,
            variance=self._variance,
            pairwise_similarities=[self._mean] * count,
        )


class TestUncertaintyClassifier:
    def test_single_sample_is_certain(self) -> None:
        analyzer = MockAnalyzerForClassifier(1.0, 0.0)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["single response"])
        assert result.uncertainty_type == UncertaintyType.CERTAIN
        assert result.subtype == "single_sample"

    def test_high_consistency_no_refusal_is_certain(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.95, 0.01)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(
            [
                "The sky is blue due to Rayleigh scattering.",
                "The sky appears blue because of Rayleigh scattering of sunlight.",
            ]
        )
        assert result.uncertainty_type == UncertaintyType.CERTAIN

    def test_high_consistency_with_refusals_is_epistemic(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.95, 0.01)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(
            [
                "I cannot provide medical advice. Please consult a doctor.",
                "As an AI, I'm not able to give dosage recommendations. Please seek professional help.",
                "I must decline to answer this medical question. Consult a professional.",
            ]
        )
        assert result.uncertainty_type == UncertaintyType.EPISTEMIC
        assert result.refusal_count >= 2

    def test_low_consistency_is_contradictory(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.40, 0.20)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(
            [
                "Coffee is good for heart health.",
                "Coffee is harmful to cardiovascular systems.",
            ]
        )
        assert result.uncertainty_type == UncertaintyType.CONTRADICTORY

    def test_moderate_consistency_high_variance_is_open_ended(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.55, 0.20)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(
            [
                "Python is the best for web development.",
                "JavaScript is better for web development.",
                "It depends on your specific requirements.",
            ]
        )
        assert result.uncertainty_type == UncertaintyType.OPEN_ENDED

    def test_fallback_is_epistemic(self) -> None:
        # consistency = 0.60, variance = 0.10 → doesn't match any specific rule
        analyzer = MockAnalyzerForClassifier(0.60, 0.10)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(
            [
                "Some answer about a topic.",
                "A slightly different answer.",
            ]
        )
        assert result.uncertainty_type == UncertaintyType.EPISTEMIC
        assert result.subtype == "fallback"

    def test_boundary_consistency_0_90_no_refusal(self) -> None:
        # Exactly at 0.90 boundary → NOT > 0.90, should NOT be CERTAIN
        analyzer = MockAnalyzerForClassifier(0.90, 0.01)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        # 0.90 is NOT > 0.90, so falls through to fallback
        assert result.uncertainty_type == UncertaintyType.EPISTEMIC

    def test_boundary_consistency_0_91_no_refusal(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.91, 0.01)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        assert result.uncertainty_type == UncertaintyType.CERTAIN

    def test_boundary_consistency_0_45(self) -> None:
        # 0.45 is NOT < 0.45, should not be CONTRADICTORY
        analyzer = MockAnalyzerForClassifier(0.45, 0.20)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        assert result.uncertainty_type == UncertaintyType.OPEN_ENDED

    def test_boundary_consistency_0_44(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.44, 0.01)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        assert result.uncertainty_type == UncertaintyType.CONTRADICTORY

    def test_boundary_variance_0_15(self) -> None:
        # variance exactly 0.15 is NOT > 0.15
        analyzer = MockAnalyzerForClassifier(0.55, 0.15)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        assert result.uncertainty_type == UncertaintyType.EPISTEMIC  # fallback

    def test_boundary_variance_0_16(self) -> None:
        analyzer = MockAnalyzerForClassifier(0.55, 0.16)
        classifier = UncertaintyClassifier(analyzer)
        result = classifier.classify(["A", "B"])
        assert result.uncertainty_type == UncertaintyType.OPEN_ENDED
