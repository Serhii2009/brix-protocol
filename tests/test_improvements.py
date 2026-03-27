"""Tests for BRIX v0.3.0 architectural improvements."""

from __future__ import annotations

import os
from unittest.mock import patch
from uuid import uuid4

import pytest

from brix.regulated.actions.executor import ActionExecutor
from brix.regulated.balance.tracker import BalanceTracker
from brix.regulated.console.output import print_result
from brix.regulated.core.exceptions import SamplerError
from brix.regulated.core.result import ActionTaken, StructuredResult, UncertaintyType
from brix.regulated.core.router import BrixRouter
from brix.regulated.engine.signal_index import SignalIndex, _normalize
from brix.regulated.llm.mock import MockLLMClient
from brix.regulated.output.guard import OutputGuard
from brix.regulated.output.result import OutputResult
from brix.regulated.retrieval.protocol import RetrievalResult
from brix.regulated.sampling.sampler import AdaptiveSampler
from brix.regulated.analysis.consistency import ConsistencyResult
from brix.regulated.spec.loader import load_spec_from_dict
from brix.regulated.spec.models import SamplingConfig, SpecModel


class MockAnalyzer:
    """Mock SemanticConsistencyAnalyzer for testing."""

    def __init__(self, mean_similarity: float = 0.95, variance: float = 0.01) -> None:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    force_retrieval_epistemic: bool = False,
    output_signals: list | None = None,
) -> SpecModel:
    """Build a minimal spec for testing."""
    data: dict = {
        "metadata": {
            "name": "test",
            "version": "1.0.0",
            "domain": "testing",
        },
        "circuit_breakers": [
            {
                "name": "test_cb",
                "patterns": ["lethal dose", "fatal dose"],
                "exclude_context": ["educational context"],
            },
        ],
        "risk_signals": [
            {
                "name": "uncertainty_lang",
                "patterns": ["is it true that"],
                "weight": 0.6,
                "category": "registered",
            },
        ],
        "uncertainty_types": [
            {
                "name": "epistemic",
                "action_config": {
                    "action": "force_retrieval",
                    "message_template": "Retrieval needed.",
                    "force_retrieval": force_retrieval_epistemic,
                },
            },
            {
                "name": "contradictory",
                "action_config": {
                    "action": "conflict_resolution",
                    "message_template": "Conflict detected.",
                },
            },
            {
                "name": "open_ended",
                "action_config": {
                    "action": "distribution_response",
                    "message_template": "Multiple perspectives.",
                },
            },
        ],
        "sampling_config": {
            "low_threshold": 0.40,
            "medium_threshold": 0.70,
        },
    }
    if output_signals is not None:
        data["output_signals"] = output_signals
    return load_spec_from_dict(data)


class MockRetrievalProvider:
    """Mock retrieval provider for testing."""

    def __init__(
        self,
        content: str = "Retrieved content.",
        score: float = 0.9,
        sources: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self._content = content
        self._score = score
        self._sources = sources or ["doc1.pdf", "doc2.pdf"]
        self._fail = fail
        self.called = False

    async def retrieve(self, query: str, *, max_results: int = 3) -> RetrievalResult:
        self.called = True
        if self._fail:
            raise RuntimeError("Retrieval provider failure")
        return RetrievalResult(
            content=self._content,
            score=self._score,
            sources=self._sources,
        )


# ---------------------------------------------------------------------------
# 1. response_requires_verification=True when CB fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_requires_verification_on_cb():
    spec = _make_spec()
    mock_llm = MockLLMClient(default_response="mock response")
    router = BrixRouter(llm_client=mock_llm, spec=spec, _analyzer=MockAnalyzer())
    result = await router.process("What is the lethal dose of aspirin?")
    assert result.circuit_breaker_hit is True
    assert result.response_requires_verification is True


# ---------------------------------------------------------------------------
# 2. response_requires_verification=True when EPISTEMIC (no retrieval provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_requires_verification_on_epistemic():
    spec = _make_spec()
    mock_llm = MockLLMClient(default_response="mock response")
    # Use a low-consistency analyzer to trigger EPISTEMIC
    router = BrixRouter(
        llm_client=mock_llm,
        spec=spec,
        _analyzer=MockAnalyzer(mean_similarity=0.80, variance=0.05),
    )
    result = await router.process("is it true that the earth is flat?")
    assert result.response_requires_verification is True


# ---------------------------------------------------------------------------
# 3. unverified_draft contains samples[0] when verification required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unverified_draft_contains_sample():
    spec = _make_spec()
    mock_llm = MockLLMClient(default_response="The answer is 42.")
    router = BrixRouter(llm_client=mock_llm, spec=spec, _analyzer=MockAnalyzer())
    result = await router.process("Tell me about the lethal dose")
    assert result.unverified_draft == "The answer is 42."


# ---------------------------------------------------------------------------
# 4. response field contains only template text (no [RETRIEVAL_NEEDED])
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_no_retrieval_needed_marker():
    spec = _make_spec()
    mock_llm = MockLLMClient(default_response="Some LLM output")
    router = BrixRouter(llm_client=mock_llm, spec=spec, _analyzer=MockAnalyzer())
    result = await router.process("What is the fatal dose of ibuprofen?")
    assert "[RETRIEVAL_NEEDED]" not in result.response
    assert "Retrieval needed." in result.response


# ---------------------------------------------------------------------------
# 5. ValueError raised for retrieval_score outside [0.0, 1.0]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_score_negative():
    spec = _make_spec()
    mock_llm = MockLLMClient()
    router = BrixRouter(llm_client=mock_llm, spec=spec, _analyzer=MockAnalyzer())
    with pytest.raises(ValueError, match="retrieval_score must be between"):
        await router.process("hello", retrieval_score=-0.1)


@pytest.mark.asyncio
async def test_retrieval_score_above_one():
    spec = _make_spec()
    mock_llm = MockLLMClient()
    router = BrixRouter(llm_client=mock_llm, spec=spec, _analyzer=MockAnalyzer())
    with pytest.raises(ValueError, match="retrieval_score must be between"):
        await router.process("hello", retrieval_score=1.5)


# ---------------------------------------------------------------------------
# 6. force_retrieval: true in action_config routes to epistemic handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_retrieval_from_action_config():
    spec = _make_spec(force_retrieval_epistemic=True)
    mock_llm = MockLLMClient(default_response="mock")
    executor = ActionExecutor(spec, mock_llm)
    # When uncertainty_type=EPISTEMIC, config says force_retrieval=True
    # even though sampler says force_retrieval=False
    result = await executor.execute(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        samples=["sample text"],
        query="safe query",
        force_retrieval=False,
    )
    assert result.action_taken == ActionTaken.FORCE_RETRIEVAL
    assert result.intervention_necessary is True


# ---------------------------------------------------------------------------
# 7. Partial LLM failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_llm_failure():
    call_count = 0

    class PartialFailLLM:
        async def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM timeout")
            return f"Response {call_count}"

    config = SamplingConfig(low_threshold=0.40, medium_threshold=0.70)
    sampler = AdaptiveSampler(PartialFailLLM(), config)
    result = await sampler.collect("test", risk_score=0.80, circuit_breaker_hit=False)
    assert result.partial_failure is True
    assert result.collected_count < result.sample_count
    assert len(result.samples) == result.collected_count


# ---------------------------------------------------------------------------
# 8. SamplerError raised only when ALL samples fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sampler_error_all_fail():
    class AllFailLLM:
        async def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024):
            raise RuntimeError("LLM down")

    config = SamplingConfig(low_threshold=0.40, medium_threshold=0.70)
    sampler = AdaptiveSampler(AllFailLLM(), config)
    with pytest.raises(SamplerError, match="Failed to collect samples"):
        await sampler.collect("test", risk_score=0.0, circuit_breaker_hit=False)


# ---------------------------------------------------------------------------
# 9. SamplingConfig raises ValueError when low >= medium
# ---------------------------------------------------------------------------


def test_sampling_config_invalid_thresholds():
    with pytest.raises(ValueError, match="low_threshold.*must be strictly less"):
        SamplingConfig(low_threshold=0.70, medium_threshold=0.70)


def test_sampling_config_invalid_thresholds_inverted():
    with pytest.raises(ValueError, match="low_threshold.*must be strictly less"):
        SamplingConfig(low_threshold=0.80, medium_threshold=0.40)


# ---------------------------------------------------------------------------
# 10. BalanceTracker uses injected risk_threshold
# ---------------------------------------------------------------------------


def test_balance_tracker_custom_threshold():
    tracker = BalanceTracker(risk_threshold=0.30)
    decision_id = uuid4()
    # risk_score=0.35 is above 0.30 but below default 0.40
    r, u, b = tracker.record_decision(
        decision_id=decision_id,
        intervention_necessary=True,
        circuit_breaker_hit=False,
        risk_score=0.35,
    )
    # With threshold 0.30, 0.35 is risky → TP
    assert tracker.state.tp == 1
    assert tracker.state.fp == 0


def test_balance_tracker_default_threshold():
    tracker = BalanceTracker()  # default 0.40
    decision_id = uuid4()
    r, u, b = tracker.record_decision(
        decision_id=decision_id,
        intervention_necessary=True,
        circuit_breaker_hit=False,
        risk_score=0.35,
    )
    # With threshold 0.40, 0.35 is NOT risky → FP
    assert tracker.state.fp == 1
    assert tracker.state.tp == 0


# ---------------------------------------------------------------------------
# 11. BalanceTracker._pending never exceeds MAX_PENDING
# ---------------------------------------------------------------------------


def test_balance_tracker_bounded_pending():
    tracker = BalanceTracker()
    for i in range(10_001):
        tracker.record_decision(
            decision_id=uuid4(),
            intervention_necessary=False,
            circuit_breaker_hit=False,
            risk_score=0.0,
        )
    assert len(tracker._pending) <= BalanceTracker.MAX_PENDING


# ---------------------------------------------------------------------------
# 12. BalanceTracker prints stderr warning for evicted feedback
# ---------------------------------------------------------------------------


def test_balance_tracker_evicted_feedback_warning(capsys):
    tracker = BalanceTracker()
    # Record MAX_PENDING + 1 decisions
    first_id = uuid4()
    tracker.record_decision(
        decision_id=first_id,
        intervention_necessary=False,
        circuit_breaker_hit=False,
        risk_score=0.0,
    )
    for i in range(BalanceTracker.MAX_PENDING):
        tracker.record_decision(
            decision_id=uuid4(),
            intervention_necessary=False,
            circuit_breaker_hit=False,
            risk_score=0.0,
        )
    # first_id should have been evicted
    tracker.feedback(first_id, was_intervention_necessary=True)
    captured = capsys.readouterr()
    assert "evicted" in captured.err


# ---------------------------------------------------------------------------
# 13. _write_log failure does not raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_log_failure_no_raise(tmp_path):
    spec = _make_spec()
    mock_llm = MockLLMClient(default_response="response")
    # Use a path that will fail (directory instead of file)
    log_path = tmp_path / "subdir"
    log_path.mkdir()
    log_file = log_path / "nonexistent_dir" / "log.jsonl"

    router = BrixRouter(
        llm_client=mock_llm,
        spec=spec,
        log_path=log_file,
        _analyzer=MockAnalyzer(),
    )
    # Should not raise even though log path parent doesn't exist
    result = await router.process("hello world")
    assert result.decision_id is not None


# ---------------------------------------------------------------------------
# 14. system_prompt is passed through to LLM calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_passthrough():
    received_systems: list[str | None] = []

    class TrackingLLM:
        async def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024):
            received_systems.append(system)
            return "response"

    spec = _make_spec()
    router = BrixRouter(
        llm_client=TrackingLLM(),
        spec=spec,
        system_prompt="You are a medical assistant.",
        _analyzer=MockAnalyzer(),
    )
    await router.process("hello")
    assert any(s == "You are a medical assistant." for s in received_systems)


# ---------------------------------------------------------------------------
# 15. OutputGuard returns clean result when output_signals is empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guard_empty_signals():
    spec = _make_spec(output_signals=[])
    guard = OutputGuard(spec, _analyzer=MockAnalyzer())
    result = await guard.analyze("You definitely have cancer")
    assert result.output_blocked is False
    assert result.response_safe_to_display is True


# ---------------------------------------------------------------------------
# 16. OutputGuard output_blocked=True for block-type signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guard_block_signal():
    spec = _make_spec(
        output_signals=[
            {
                "name": "definitive_diagnosis",
                "patterns": ["you have cancer", "you are diagnosed"],
                "weight": 0.9,
                "signal_type": "block",
            },
        ]
    )
    guard = OutputGuard(spec, _analyzer=MockAnalyzer())
    result = await guard.analyze("Based on your symptoms, you have cancer.")
    assert result.output_blocked is True
    assert result.output_block_signal == "definitive_diagnosis"
    assert result.response_safe_to_display is False


# ---------------------------------------------------------------------------
# 17. OutputGuard output_blocked=False for pattern in query not response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guard_pattern_in_query_not_response():
    spec = _make_spec(
        output_signals=[
            {
                "name": "definitive_diagnosis",
                "patterns": ["you have cancer"],
                "weight": 0.9,
                "signal_type": "block",
            },
        ]
    )
    guard = OutputGuard(spec, _analyzer=MockAnalyzer())
    # Pattern is in query, but response is clean
    result = await guard.analyze(
        "The treatment plan is standard chemotherapy.",
        query="Does the patient you have cancer?",
    )
    assert result.output_blocked is False


# ---------------------------------------------------------------------------
# 18. RetrievalProvider called when force_retrieval and provider configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_provider_called():
    spec = _make_spec()
    provider = MockRetrievalProvider(content="Aspirin LD50 is 200mg/kg")
    mock_llm = MockLLMClient(default_response="mock")
    executor = ActionExecutor(spec, mock_llm, retrieval_provider=provider)
    result = await executor.execute(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        samples=["mock"],
        query="What is the lethal dose of aspirin?",
    )
    assert provider.called is True
    assert result.retrieval_executed is True
    assert result.retrieval_failed is False
    assert "Aspirin LD50" in result.response
    assert "doc1.pdf" in result.retrieval_sources


# ---------------------------------------------------------------------------
# 19. RetrievalProvider failure: graceful fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_provider_failure():
    spec = _make_spec()
    provider = MockRetrievalProvider(fail=True)
    mock_llm = MockLLMClient(default_response="mock")
    executor = ActionExecutor(spec, mock_llm, retrieval_provider=provider)
    result = await executor.execute(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        samples=["draft response"],
        query="test query",
    )
    assert provider.called is True
    assert result.retrieval_failed is True
    assert result.response_requires_verification is True
    assert result.unverified_draft == "draft response"


# ---------------------------------------------------------------------------
# 20. print_result() is no-op when stdout is not a TTY and BRIX_CONSOLE unset
# ---------------------------------------------------------------------------


def test_print_result_noop_non_tty():
    result = StructuredResult(
        uncertainty_type=UncertaintyType.CERTAIN,
        action_taken=ActionTaken.NONE,
        response="test",
        circuit_breaker_hit=False,
        risk_score=0.0,
        reliability_signal=True,
        utility_signal=True,
        balance_index=0.0,
        intervention_necessary=False,
        registry_version="test/1.0",
        latency_ms=10.0,
    )
    with patch.dict(os.environ, {}, clear=True):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            # Should not raise, should be a no-op
            print_result(result)


# ---------------------------------------------------------------------------
# 21. print_result() failure never propagates
# ---------------------------------------------------------------------------


def test_print_result_failure_no_propagate():
    result = StructuredResult(
        uncertainty_type=UncertaintyType.CERTAIN,
        action_taken=ActionTaken.NONE,
        response="test",
        circuit_breaker_hit=False,
        risk_score=0.0,
        reliability_signal=True,
        utility_signal=True,
        balance_index=0.0,
        intervention_necessary=False,
        registry_version="test/1.0",
        latency_ms=10.0,
    )
    with patch.dict(os.environ, {"BRIX_CONSOLE": "1"}):
        with patch("rich.console.Console", side_effect=RuntimeError("boom")):
            # Should not raise
            print_result(result)


# ---------------------------------------------------------------------------
# 22. Unicode normalization: non-breaking space matches pattern
# ---------------------------------------------------------------------------


def test_normalize_non_breaking_space():
    spec = _make_spec()
    index = SignalIndex(spec)
    # Use non-breaking space between "lethal" and "dose"
    matches = index.scan("What is the lethal\u00a0dose?")
    cb_matches = [m for m in matches if m.signal_name == "test_cb"]
    assert len(cb_matches) > 0


def test_normalize_function():
    assert _normalize("hello\u00a0world") == "hello world"
    assert _normalize("  double   space  ") == "double space"
    assert _normalize("line\nbreak") == "line break"


# ---------------------------------------------------------------------------
# 23. SignalIndex._build() correct matches (no mutation bugs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Additional coverage: console output with BRIX_CONSOLE=1
# ---------------------------------------------------------------------------


def test_print_result_enabled_via_env():
    result = StructuredResult(
        uncertainty_type=UncertaintyType.CERTAIN,
        action_taken=ActionTaken.NONE,
        response="test",
        circuit_breaker_hit=False,
        risk_score=0.0,
        reliability_signal=True,
        utility_signal=True,
        balance_index=0.5,
        intervention_necessary=False,
        registry_version="test/1.0",
        latency_ms=10.0,
    )
    with patch.dict(os.environ, {"BRIX_CONSOLE": "1", "BRIX_VERBOSE": "1"}):
        # Should not raise, exercises the full output path
        print_result(result)


def test_print_result_blocked_with_output():
    result = StructuredResult(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        action_taken=ActionTaken.FORCE_RETRIEVAL,
        response="blocked",
        circuit_breaker_hit=True,
        circuit_breaker_name="test_cb",
        risk_score=1.0,
        reliability_signal=True,
        utility_signal=True,
        balance_index=0.5,
        intervention_necessary=True,
        registry_version="test/1.0",
        latency_ms=50.0,
        retrieval_executed=True,
        retrieval_sources=["doc1.pdf"],
    )
    output = OutputResult(
        output_blocked=True,
        output_block_signal="diagnosis",
        output_risk_score=1.0,
    )
    with patch.dict(os.environ, {"BRIX_CONSOLE": "1", "BRIX_VERBOSE": "1"}):
        print_result(result, output_result=output)


def test_print_result_elevated():
    result = StructuredResult(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        action_taken=ActionTaken.FORCE_RETRIEVAL,
        response="needs retrieval",
        circuit_breaker_hit=False,
        risk_score=0.6,
        reliability_signal=True,
        utility_signal=True,
        balance_index=0.8,
        intervention_necessary=True,
        registry_version="test/1.0",
        latency_ms=80.0,
        retrieval_failed=True,
    )
    with patch.dict(os.environ, {"BRIX_CONSOLE": "1", "BRIX_VERBOSE": "1"}):
        print_result(result)


# ---------------------------------------------------------------------------
# Additional coverage: output analyzer with universal category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guard_risk_signal():
    spec = _make_spec(
        output_signals=[
            {
                "name": "return_claim",
                "patterns": ["guaranteed return"],
                "weight": 0.8,
                "category": "registered",
                "signal_type": "risk",
            },
            {
                "name": "generic_risk",
                "patterns": ["invest now"],
                "weight": 0.3,
                "category": "universal",
                "signal_type": "risk",
            },
        ]
    )
    guard = OutputGuard(spec, _analyzer=MockAnalyzer())
    result = await guard.analyze("This investment offers a guaranteed return. Invest now!")
    assert result.output_blocked is False
    assert result.output_risk_score > 0.0
    assert len(result.output_signals_triggered) == 2


# ---------------------------------------------------------------------------
# Additional coverage: output analyzer exclude_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guard_exclude_context():
    spec = _make_spec(
        output_signals=[
            {
                "name": "diagnosis",
                "patterns": ["you have"],
                "weight": 0.9,
                "signal_type": "block",
                "exclude_context": ["educational"],
            },
        ]
    )
    guard = OutputGuard(spec, _analyzer=MockAnalyzer())
    result = await guard.analyze(
        "you have a condition",
        query="In an educational setting, what conditions exist?",
    )
    assert result.output_blocked is False


def test_signal_index_duplicate_patterns():
    """Two signals sharing the same normalized pattern both appear in results."""
    data = {
        "metadata": {"name": "dup", "version": "1.0", "domain": "test"},
        "circuit_breakers": [
            {"name": "cb1", "patterns": ["test pattern"]},
        ],
        "risk_signals": [
            {"name": "rs1", "patterns": ["test pattern"], "weight": 0.5},
        ],
    }
    spec = load_spec_from_dict(data)
    index = SignalIndex(spec)
    matches = index.scan("this has test pattern in it")
    names = {m.signal_name for m in matches}
    assert "cb1" in names
    assert "rs1" in names
