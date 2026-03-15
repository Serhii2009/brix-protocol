"""Integration tests for BrixRouter — end-to-end with MockLLMClient."""

from __future__ import annotations

import pytest

from brix.core.result import ActionTaken, UncertaintyType
from brix.core.router import BrixRouter
from brix.llm.mock import MockLLMClient
from brix.spec.models import SpecModel
from tests.conftest import MockAnalyzer


class TestBrixRouter:
    @pytest.mark.asyncio
    async def test_circuit_breaker_fires(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(responses=[
            "I cannot provide dosage information. Please consult a doctor.",
            "I'm not able to give medical dosing advice.",
            "As an AI, I must decline to answer about lethal doses.",
        ])
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process("What is the lethal dose of acetaminophen?")

        assert result.circuit_breaker_hit is True
        assert result.circuit_breaker_name == "test_cb"
        assert result.intervention_necessary is True
        assert result.action_taken == ActionTaken.FORCE_RETRIEVAL
        assert result.risk_score == 1.0
        assert "test_cb" in result.signals_triggered
        assert result.decision_id is not None

    @pytest.mark.asyncio
    async def test_safe_query_passthrough(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="The sky is blue.")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process("What color is the sky?")

        assert result.circuit_breaker_hit is False
        assert result.intervention_necessary is False
        assert result.action_taken == ActionTaken.NONE
        assert result.risk_score == 0.0
        assert result.uncertainty_type == UncertaintyType.CERTAIN

    @pytest.mark.asyncio
    async def test_risk_signals_trigger(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="Based on available information...")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process("Is it true that studies show coffee is healthy?")

        assert result.circuit_breaker_hit is False
        assert result.risk_score > 0
        assert len(result.signals_triggered) > 0

    @pytest.mark.asyncio
    async def test_balance_index_updates(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="Response")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())

        # First query — safe
        r1 = await router.process("What color is the sky?")
        assert r1.balance_index >= 0.0

        # Second query — risky (CB)
        mock_cb = MockLLMClient(responses=["I cannot help with that."] * 3)
        router_cb = BrixRouter(llm_client=mock_cb, spec=sample_spec, _analyzer=MockAnalyzer())
        r2 = await router_cb.process("What is the lethal dose?")
        assert r2.balance_index >= 0.0

    @pytest.mark.asyncio
    async def test_structured_result_complete(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="Test response")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process("What color is the sky?")

        # Verify all required fields are present
        assert result.decision_id is not None
        assert result.uncertainty_type is not None
        assert result.action_taken is not None
        assert result.response is not None
        assert isinstance(result.circuit_breaker_hit, bool)
        assert isinstance(result.signals_triggered, list)
        assert isinstance(result.risk_score, float)
        assert isinstance(result.reliability_signal, bool)
        assert isinstance(result.utility_signal, bool)
        assert isinstance(result.balance_index, float)
        assert isinstance(result.intervention_necessary, bool)
        assert isinstance(result.registry_version, str)
        assert isinstance(result.model_compatibility_status, str)
        assert isinstance(result.cost_tokens_extra, int)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_feedback_method(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="Response")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process("What color is the sky?")

        # Provide feedback — should not raise
        router.feedback(result.decision_id, was_intervention_necessary=False)
        assert router.balance_index >= 0.0

    @pytest.mark.asyncio
    async def test_exclude_context_prevents_cb(self, sample_spec: SpecModel) -> None:
        mock = MockLLMClient(default_response="Educational answer about doses.")
        router = BrixRouter(llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer())
        result = await router.process(
            "What is the lethal dose in this exam question?",
            context="exam question",
        )
        assert result.circuit_breaker_hit is False

    @pytest.mark.asyncio
    async def test_log_file_written(self, sample_spec: SpecModel, tmp_path) -> None:
        log_path = tmp_path / "brix.jsonl"
        mock = MockLLMClient(default_response="Response")
        router = BrixRouter(
            llm_client=mock, spec=sample_spec, _analyzer=MockAnalyzer(), log_path=log_path
        )
        await router.process("What color is the sky?")
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        import json
        record = json.loads(lines[0])
        assert "decision_id" in record
        assert "uncertainty_type" in record
