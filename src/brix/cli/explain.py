"""brix explain — reconstruct a complete decision trace from structured result logs."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def explain_cmd(
    decision_id: str = typer.Option(..., "--decision-id", "-d", help="UUID of the decision to explain"),
    log: str = typer.Option(..., "--log", "-l", help="Path to the JSONL log file"),
) -> None:
    """Reconstruct a complete decision trace for audit and debugging."""
    console.print(f"\n[bold]BRIX Explain[/bold] — Decision {decision_id}\n")

    log_path = Path(log)
    if not log_path.exists():
        console.print(f"[red]ERROR[/red] Log file not found: {log_path}")
        raise typer.Exit(code=2)

    # Search for the decision in the JSONL log
    record = _find_decision(log_path, decision_id)
    if record is None:
        console.print(f"[red]ERROR[/red] Decision {decision_id} not found in log")
        raise typer.Exit(code=1)

    # Display the complete decision trace
    _display_trace(record)
    raise typer.Exit(code=0)


def _find_decision(log_path: Path, decision_id: str) -> dict | None:
    """Search a JSONL log file for a specific decision ID."""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("decision_id", "")) == decision_id:
                return record
    return None


def _display_trace(record: dict) -> None:
    """Display the full decision trace from a structured result record."""
    # Header
    console.print(Panel(
        f"Decision ID: {record.get('decision_id', 'N/A')}",
        title="Decision Trace",
        style="bold blue",
    ))

    # Signal evaluation
    signals_table = Table(title="Signal Evaluation")
    signals_table.add_column("Signal", style="bold")
    signals_table.add_column("Triggered")

    signals = record.get("signals_triggered", [])
    if signals:
        for sig in signals:
            signals_table.add_row(sig, "[red]YES[/red]")
    else:
        signals_table.add_row("(none)", "[green]NO[/green]")
    console.print(signals_table)

    # Circuit breaker status
    cb_hit = record.get("circuit_breaker_hit", False)
    cb_name = record.get("circuit_breaker_name", "N/A")
    if cb_hit:
        console.print(f"\n[red bold]Circuit Breaker FIRED[/red bold]: {cb_name}")
    else:
        console.print(f"\n[green]Circuit Breaker: Not triggered[/green]")

    # Risk score breakdown
    risk_table = Table(title="Risk Assessment")
    risk_table.add_column("Component", style="bold")
    risk_table.add_column("Value")
    risk_table.add_row("Risk Score", f"{record.get('risk_score', 0.0):.4f}")
    risk_table.add_row("Circuit Breaker Hit", str(cb_hit))
    console.print(risk_table)

    # Uncertainty classification
    class_table = Table(title="Uncertainty Classification")
    class_table.add_column("Field", style="bold")
    class_table.add_column("Value")
    class_table.add_row("Uncertainty Type", record.get("uncertainty_type", "N/A"))
    class_table.add_row("Subtype", record.get("subtype", "N/A"))
    class_table.add_row("Action Taken", record.get("action_taken", "N/A"))
    class_table.add_row("Intervention Necessary", str(record.get("intervention_necessary", False)))
    console.print(class_table)

    # Metrics
    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value")
    metrics_table.add_row("Balance Index", f"{record.get('balance_index', 0.0):.4f}")
    metrics_table.add_row("Reliability Signal", str(record.get("reliability_signal", False)))
    metrics_table.add_row("Utility Signal", str(record.get("utility_signal", False)))
    metrics_table.add_row("Extra Tokens", str(record.get("cost_tokens_extra", 0)))
    metrics_table.add_row("Latency (ms)", f"{record.get('latency_ms', 0.0):.2f}")
    metrics_table.add_row("Registry Version", record.get("registry_version", "N/A"))
    metrics_table.add_row("Model Compatibility", record.get("model_compatibility_status", "N/A"))
    console.print(metrics_table)

    # Response preview
    response = record.get("response", "")
    if len(response) > 500:
        response = response[:500] + "..."
    console.print(Panel(response, title="Response", style="dim"))
