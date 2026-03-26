"""OutputAnalyzer — scans LLM responses using the same Aho-Corasick infrastructure."""

from __future__ import annotations

from brix.regulated.engine.signal_index import SignalIndex, _normalize
from brix.regulated.output.result import OutputResult
from brix.regulated.spec.models import (
    Metadata,
    OutputSignalDef,
    RiskSignalDef,
    SpecModel,
)


class OutputAnalyzer:
    """Analyzes LLM responses for output-side risk signals.

    Builds a dedicated SignalIndex from output_signals defined in the spec,
    reusing all existing Aho-Corasick infrastructure without modification.
    """

    def __init__(self, spec: SpecModel) -> None:
        self._output_signals = spec.output_signals
        self._signal_index: SignalIndex | None = None
        self._weight_map: dict[str, float] = {}
        self._category_map: dict[str, str] = {}
        self._signal_type_map: dict[str, str] = {}
        self._exclude_map: dict[str, list[str]] = {}

        if self._output_signals:
            self._build_index()

    def _build_index(self) -> None:
        """Build a dedicated SignalIndex from output signals."""
        # Map output signals to RiskSignalDef for the synthetic SpecModel
        risk_signal_defs: list[RiskSignalDef] = []
        for sig in self._output_signals:
            risk_signal_defs.append(
                RiskSignalDef(
                    name=sig.name,
                    patterns=sig.patterns,
                    weight=sig.weight,
                    category=sig.category,
                    description=sig.description,
                    exclude_context=sig.exclude_context,
                )
            )
            self._weight_map[sig.name] = sig.weight
            self._category_map[sig.name] = sig.category
            self._signal_type_map[sig.name] = sig.signal_type
            self._exclude_map[sig.name] = [
                _normalize(p).lower() for p in sig.exclude_context
            ]

        synthetic_spec = SpecModel(
            metadata=Metadata(
                name="_output_internal",
                version="0",
                domain="_internal",
            ),
            risk_signals=risk_signal_defs,
        )
        self._signal_index = SignalIndex(synthetic_spec)

    def analyze(
        self,
        response: str,
        *,
        query: str | None = None,
        context: str | None = None,
    ) -> OutputResult:
        """Analyze a response for output-side risk signals.

        Args:
            response: The LLM response text to scan.
            query: Original query (used for exclude_context filtering).
            context: Optional context (used for exclude_context filtering).

        Returns:
            OutputResult with blocking status and risk score.
        """
        if not self._output_signals or self._signal_index is None:
            return OutputResult()

        # Scan the response text
        matches = self._signal_index.scan(response)
        risk_matches = [m for m in matches if m.signal_type == "risk_signal"]

        # Apply exclude_context filtering using query+context (not response)
        combined_text = ""
        if query or context:
            combined_text = _normalize(f"{query or ''} {context or ''}").lower()

        surviving_names: set[str] = set()
        for match in risk_matches:
            exclusions = self._exclude_map.get(match.signal_name, [])
            cancelled = any(exc in combined_text for exc in exclusions)
            if not cancelled:
                surviving_names.add(match.signal_name)

        if not surviving_names:
            return OutputResult()

        # Separate block signals from risk signals
        block_signals: list[str] = []
        risk_signal_names: list[str] = []
        for name in surviving_names:
            if self._signal_type_map.get(name) == "block":
                block_signals.append(name)
            else:
                risk_signal_names.append(name)

        # Compute output risk score using same formula as RiskScoreTrack
        registered_weights: list[float] = []
        universal_weights: list[float] = []
        for name in risk_signal_names:
            weight = self._weight_map.get(name, 0.0)
            category = self._category_map.get(name, "registered")
            if category == "universal":
                universal_weights.append(weight)
            else:
                registered_weights.append(weight)

        max_registered = max(registered_weights) if registered_weights else 0.0
        sum_universal = sum(universal_weights)
        registered_component = max_registered * 1.0
        universal_component = sum_universal * 0.6
        raw_score = registered_component + universal_component
        output_risk_score = min(max(raw_score, 0.0), 1.0)

        # Block signals override everything
        output_blocked = len(block_signals) > 0
        if output_blocked:
            output_risk_score = 1.0

        return OutputResult(
            output_blocked=output_blocked,
            output_risk_score=output_risk_score,
            output_signals_triggered=sorted(surviving_names),
            output_block_signal=block_signals[0] if block_signals else None,
            output_risk_breakdown={
                "max_registered": max_registered,
                "registered_component": registered_component,
                "sum_universal": sum_universal,
                "universal_component": universal_component,
                "raw_score": raw_score,
            },
            response_safe_to_display=not output_blocked and output_risk_score < 0.70,
        )
