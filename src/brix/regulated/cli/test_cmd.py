"""brix test — run a test suite against a specification and report scores."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from brix.regulated.core.exceptions import SpecValidationError
from brix.regulated.core.router import BrixRouter
from brix.regulated.llm.mock import MockLLMClient
from brix.regulated.spec.loader import load_spec

console = Console()


def test_cmd(
    spec_file: str = typer.Argument(help="Path to the uncertainty.yaml specification"),
    model: str = typer.Option(
        "mock", "--model", "-m", help="Model identifier (use 'mock' for testing)"
    ),
    suite: str | None = typer.Option(None, "--suite", "-s", help="Path to test suite YAML file"),
) -> None:
    """Run a test suite against a BRIX specification."""
    console.print(f"\n[bold]BRIX Test[/bold] — {Path(spec_file).name}\n")

    # Load spec
    try:
        spec = load_spec(spec_file)
    except SpecValidationError as exc:
        console.print(f"[red]ERROR[/red] Failed to load spec: {exc}")
        raise typer.Exit(code=2)

    # Load test suite
    if suite is None:
        console.print(
            "[yellow]No test suite provided. Use --suite to specify a test file.[/yellow]"
        )
        console.print("Generate a test suite with: brix generate-tests <spec_file> --output <dir>")
        raise typer.Exit(code=1)

    suite_path = Path(suite)
    if not suite_path.exists():
        console.print(f"[red]ERROR[/red] Test suite file not found: {suite_path}")
        raise typer.Exit(code=2)

    try:
        suite_data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        console.print(f"[red]ERROR[/red] Invalid test suite YAML: {exc}")
        raise typer.Exit(code=2)

    test_cases = suite_data.get("test_cases", [])
    if not test_cases:
        console.print("[yellow]No test cases found in suite[/yellow]")
        raise typer.Exit(code=1)

    # Note about model parameter
    if model != "mock":
        console.print(
            f"[yellow]Note:[/yellow] brix test always uses MockLLMClient for deterministic results. "
            f"The --model '{model}' value is recorded in the report but does not affect test execution. "
            f"Declare model compatibility in spec metadata.model_compatibility instead."
        )

    # Create router with mock client
    mock_client = MockLLMClient(default_response="Mock response for testing.")
    router = BrixRouter(llm_client=mock_client, spec=spec, _analyzer=_create_mock_analyzer())

    # Run tests
    results = asyncio.run(_run_tests(router, test_cases))

    # Compute scores
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    tp = sum(1 for r in results if r["expected_intervention"] and r["actual_intervention"])
    fn = sum(1 for r in results if r["expected_intervention"] and not r["actual_intervention"])
    tn = sum(1 for r in results if not r["expected_intervention"] and not r["actual_intervention"])
    fp = sum(1 for r in results if not r["expected_intervention"] and r["actual_intervention"])

    reliability = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    utility = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balance = (
        2.0 * reliability * utility / (reliability + utility)
        if (reliability + utility) > 0
        else 0.0
    )

    # Display results
    table = Table(title="Test Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total Cases", str(len(results)))
    table.add_row("Passed", f"[green]{passed}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    table.add_row("Reliability Score", f"{reliability:.3f}")
    table.add_row("Utility Score", f"{utility:.3f}")
    table.add_row("Balance Index", f"{balance:.3f}")
    table.add_row("TP / FN / TN / FP", f"{tp} / {fn} / {tn} / {fp}")
    console.print(table)

    # Show failing cases
    if failed > 0:
        console.print("\n[red]Failing cases:[/red]")
        for r in results:
            if not r["passed"]:
                console.print(f"  • [bold]{r['name']}[/bold]")
                console.print(f"    Query: {r['query']}")
                console.print(f"    Expected intervention: {r['expected_intervention']}")
                console.print(f"    Actual intervention: {r['actual_intervention']}")
                console.print()

    # JSON report to stdout
    report = {
        "spec": f"{spec.metadata.name}/{spec.metadata.version}",
        "model": model,
        "target_model": model,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "reliability_score": reliability,
        "utility_score": utility,
        "balance_index": balance,
        "confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }
    console.print("\n[dim]JSON Report:[/dim]")
    console.print(json.dumps(report, indent=2))

    if failed > 0:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


async def _run_tests(router: BrixRouter, test_cases: list[dict]) -> list[dict]:
    """Run all test cases and collect results."""
    results: list[dict] = []
    for case in test_cases:
        query = case.get("query", "")
        expected_intervention = case.get("expected_intervention", False)
        expected_cb = case.get("expected_circuit_breaker", False)
        name = case.get("name", query[:50])

        result = await router.process(query=query, context=case.get("context"))

        actual_intervention = result.intervention_necessary
        passed = actual_intervention == expected_intervention
        if expected_cb and not result.circuit_breaker_hit:
            passed = False

        results.append(
            {
                "name": name,
                "query": query,
                "expected_intervention": expected_intervention,
                "actual_intervention": actual_intervention,
                "expected_cb": expected_cb,
                "actual_cb": result.circuit_breaker_hit,
                "passed": passed,
                "risk_score": result.risk_score,
                "uncertainty_type": result.uncertainty_type,
                "signals": result.signals_triggered,
            }
        )

    return results


def _create_mock_analyzer():
    """Create a mock analyzer that skips sentence-transformers loading."""
    from brix.regulated.analysis.consistency import ConsistencyResult

    class MockAnalyzer:
        def analyze(self, samples: list[str]) -> ConsistencyResult:
            return ConsistencyResult(
                mean_similarity=0.95,
                variance=0.01,
                pairwise_similarities=[0.95],
            )

    return MockAnalyzer()
