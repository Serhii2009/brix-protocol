"""Tests for StructuredResult model."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from brix.core.result import ActionTaken, StructuredResult, UncertaintyType


class TestStructuredResult:
    def test_all_fields_present(self) -> None:
        result = StructuredResult(
            decision_id=uuid4(),
            uncertainty_type=UncertaintyType.CERTAIN,
            subtype="test",
            action_taken=ActionTaken.NONE,
            response="Test response",
            circuit_breaker_hit=False,
            circuit_breaker_name=None,
            signals_triggered=[],
            risk_score=0.0,
            reliability_signal=True,
            utility_signal=True,
            balance_index=0.0,
            intervention_necessary=False,
            registry_version="test/1.0.0",
            model_compatibility_status="unknown",
            cost_tokens_extra=0,
            latency_ms=1.5,
        )
        assert isinstance(result.decision_id, UUID)
        assert result.uncertainty_type == UncertaintyType.CERTAIN
        assert result.action_taken == ActionTaken.NONE
        assert result.circuit_breaker_hit is False
        assert result.risk_score == 0.0
        assert result.intervention_necessary is False

    def test_uuid_auto_generated(self) -> None:
        result = StructuredResult(
            uncertainty_type=UncertaintyType.EPISTEMIC,
            action_taken=ActionTaken.FORCE_RETRIEVAL,
            response="test",
            circuit_breaker_hit=True,
            signals_triggered=["sig1"],
            risk_score=1.0,
            reliability_signal=True,
            utility_signal=False,
            balance_index=0.5,
            intervention_necessary=True,
            registry_version="test/1.0.0",
            cost_tokens_extra=100,
            latency_ms=50.0,
        )
        assert isinstance(result.decision_id, UUID)

    def test_serializes_to_json(self) -> None:
        result = StructuredResult(
            uncertainty_type=UncertaintyType.CONTRADICTORY,
            action_taken=ActionTaken.CONFLICT_RESOLUTION,
            response="conflict response",
            circuit_breaker_hit=False,
            signals_triggered=["sig_a", "sig_b"],
            risk_score=0.75,
            reliability_signal=True,
            utility_signal=True,
            balance_index=0.85,
            intervention_necessary=True,
            registry_version="general/1.0.0",
            model_compatibility_status="community",
            cost_tokens_extra=200,
            latency_ms=120.5,
        )
        json_str = result.model_dump_json()
        data = json.loads(json_str)
        assert data["uncertainty_type"] == "contradictory"
        assert data["action_taken"] == "conflict_resolution"
        assert data["risk_score"] == 0.75
        assert len(data["signals_triggered"]) == 2

    def test_enum_serialization(self) -> None:
        result = StructuredResult(
            uncertainty_type=UncertaintyType.OPEN_ENDED,
            action_taken=ActionTaken.DISTRIBUTION_RESPONSE,
            response="distribution",
            circuit_breaker_hit=False,
            signals_triggered=[],
            risk_score=0.5,
            reliability_signal=True,
            utility_signal=True,
            balance_index=0.7,
            intervention_necessary=True,
            registry_version="test/1.0.0",
            cost_tokens_extra=0,
            latency_ms=10.0,
        )
        data = result.model_dump()
        assert data["uncertainty_type"] == "open_ended"
        assert data["action_taken"] == "distribution_response"

    def test_model_compatibility_status_values(self) -> None:
        for status in ["verified", "community", "untested", "unknown"]:
            result = StructuredResult(
                uncertainty_type=UncertaintyType.CERTAIN,
                action_taken=ActionTaken.NONE,
                response="test",
                circuit_breaker_hit=False,
                signals_triggered=[],
                risk_score=0.0,
                reliability_signal=True,
                utility_signal=True,
                balance_index=0.0,
                intervention_necessary=False,
                registry_version="test/1.0.0",
                model_compatibility_status=status,
                cost_tokens_extra=0,
                latency_ms=0.0,
            )
            assert result.model_compatibility_status == status
