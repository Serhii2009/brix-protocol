"""Uncertainty type classifier using semantic consistency and refusal signals.

Classification thresholds:
  - consistency > 0.90, no refusal signals     → CERTAIN
  - consistency > 0.90, refusal in ≥2 samples  → EPISTEMIC
  - consistency < 0.45                         → CONTRADICTORY
  - 0.45 ≤ consistency < 0.70, variance > 0.15 → OPEN_ENDED
  - all other cases                            → EPISTEMIC (safe fallback)
"""

from __future__ import annotations

from dataclasses import dataclass

from brix.analysis.consistency import ConsistencyResult, SemanticConsistencyAnalyzer
from brix.analysis.refusal import count_refusals
from brix.core.result import UncertaintyType


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of uncertainty type classification."""

    uncertainty_type: UncertaintyType
    subtype: str
    mean_consistency: float
    variance: float
    refusal_count: int


class UncertaintyClassifier:
    """Classifies uncertainty type from response samples.

    Uses the SemanticConsistencyAnalyzer for embedding-based similarity
    and refusal detection for behavioral signals. Applies fixed thresholds
    to produce a deterministic classification given the same inputs.
    """

    def __init__(self, analyzer: SemanticConsistencyAnalyzer) -> None:
        self._analyzer = analyzer

    def classify(self, samples: list[str]) -> ClassificationResult:
        """Classify the uncertainty type from collected samples.

        Args:
            samples: List of LLM response texts.

        Returns:
            ClassificationResult with the determined uncertainty type.
        """
        # Single sample → CERTAIN (low-risk passthrough)
        if len(samples) <= 1:
            return ClassificationResult(
                uncertainty_type=UncertaintyType.CERTAIN,
                subtype="single_sample",
                mean_consistency=1.0,
                variance=0.0,
                refusal_count=0,
            )

        consistency = self._analyzer.analyze(samples)
        refusal_count = count_refusals(samples)

        uncertainty_type, subtype = self._apply_thresholds(
            consistency, refusal_count
        )

        return ClassificationResult(
            uncertainty_type=uncertainty_type,
            subtype=subtype,
            mean_consistency=consistency.mean_similarity,
            variance=consistency.variance,
            refusal_count=refusal_count,
        )

    def _apply_thresholds(
        self,
        consistency: ConsistencyResult,
        refusal_count: int,
    ) -> tuple[UncertaintyType, str]:
        """Apply classification thresholds to consistency and refusal data.

        Returns:
            Tuple of (UncertaintyType, subtype_string).
        """
        sim = consistency.mean_similarity
        var = consistency.variance

        # High consistency, no refusals → CERTAIN
        if sim > 0.90 and refusal_count == 0:
            return UncertaintyType.CERTAIN, "high_consistency_no_refusal"

        # High consistency, but refusals present → EPISTEMIC
        if sim > 0.90 and refusal_count >= 2:
            return UncertaintyType.EPISTEMIC, "high_consistency_with_refusals"

        # Very low consistency → CONTRADICTORY
        if sim < 0.45:
            return UncertaintyType.CONTRADICTORY, "low_consistency"

        # Moderate consistency with high variance → OPEN_ENDED
        if 0.45 <= sim < 0.70 and var > 0.15:
            return UncertaintyType.OPEN_ENDED, "moderate_consistency_high_variance"

        # Safe fallback → EPISTEMIC
        return UncertaintyType.EPISTEMIC, "fallback"
