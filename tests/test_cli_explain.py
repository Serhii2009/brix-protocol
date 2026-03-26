"""Tests for brix explain CLI command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from brix.regulated.cli.main import app

runner = CliRunner()


class TestExplainCommand:
    def test_explain_found(self, tmp_path) -> None:
        log_file = tmp_path / "brix.jsonl"
        record = {
            "decision_id": "550e8400-e29b-41d4-a716-446655440000",
            "uncertainty_type": "epistemic",
            "subtype": "fallback",
            "action_taken": "force_retrieval",
            "response": "Test response",
            "circuit_breaker_hit": True,
            "circuit_breaker_name": "medical_dosing",
            "signals_triggered": ["medical_dosing"],
            "risk_score": 1.0,
            "reliability_signal": True,
            "utility_signal": True,
            "balance_index": 0.85,
            "intervention_necessary": True,
            "registry_version": "general/1.0.0",
            "model_compatibility_status": "community",
            "cost_tokens_extra": 150,
            "latency_ms": 45.2,
        }
        log_file.write_text(json.dumps(record) + "\n")

        result = runner.invoke(app, [
            "explain",
            "--decision-id", "550e8400-e29b-41d4-a716-446655440000",
            "--log", str(log_file),
        ])
        assert result.exit_code == 0
        assert "medical_dosing" in result.output
        assert "FIRED" in result.output

    def test_explain_not_found(self, tmp_path) -> None:
        log_file = tmp_path / "brix.jsonl"
        log_file.write_text('{"decision_id": "other-id"}\n')

        result = runner.invoke(app, [
            "explain",
            "--decision-id", "550e8400-e29b-41d4-a716-446655440000",
            "--log", str(log_file),
        ])
        assert result.exit_code == 1

    def test_explain_missing_log(self) -> None:
        result = runner.invoke(app, [
            "explain",
            "--decision-id", "550e8400-e29b-41d4-a716-446655440000",
            "--log", "/nonexistent/path.jsonl",
        ])
        assert result.exit_code == 2
