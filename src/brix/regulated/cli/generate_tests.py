"""brix generate-tests — generate a draft test suite from a specification."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from brix.regulated.core.exceptions import SpecValidationError
from brix.regulated.spec.loader import load_spec

console = Console()


def generate_tests_cmd(
    spec_file: str = typer.Argument(help="Path to the uncertainty.yaml specification"),
    output: str = typer.Option(
        "generated_tests", "--output", "-o", help="Output directory for generated tests"
    ),
) -> None:
    """Generate a draft test suite from a BRIX specification."""
    console.print(f"\n[bold]BRIX Generate Tests[/bold] — {Path(spec_file).name}\n")

    # Load spec
    try:
        spec = load_spec(spec_file)
    except SpecValidationError as exc:
        console.print(f"[red]ERROR[/red] Failed to load spec: {exc}")
        raise typer.Exit(code=2)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases: list[dict] = []
    counts = {
        "circuit_breaker_positive": 0,
        "circuit_breaker_negative": 0,
        "risk_signal": 0,
        "uncertainty_type": 0,
    }

    # Generate circuit breaker positive cases
    for cb in spec.circuit_breakers:
        for pattern in cb.patterns:
            test_cases.append(
                {
                    "name": f"cb_positive_{cb.name}_{_slugify(pattern)}",
                    "category": "circuit_breaker_positive",
                    "query": f"Tell me about {pattern} in detail",
                    "expected_intervention": True,
                    "expected_circuit_breaker": True,
                    "expected_breaker_name": cb.name,
                    "status": "draft",
                }
            )
            counts["circuit_breaker_positive"] += 1

        # Generate negative cases using exclude_context
        for exclusion in cb.exclude_context:
            test_cases.append(
                {
                    "name": f"cb_negative_{cb.name}_{_slugify(exclusion)}",
                    "category": "circuit_breaker_negative",
                    "query": f"In a {exclusion} context, explain {cb.patterns[0]}",
                    "context": exclusion,
                    "expected_intervention": False,
                    "expected_circuit_breaker": False,
                    "status": "draft",
                }
            )
            counts["circuit_breaker_negative"] += 1

    # Generate risk signal cases
    for signal in spec.risk_signals:
        for pattern in signal.patterns[:3]:  # Limit to 3 per signal to avoid bloat
            test_cases.append(
                {
                    "name": f"risk_{signal.name}_{_slugify(pattern)}",
                    "category": "risk_signal",
                    "query": f"Can you tell me {pattern} for this topic?",
                    "expected_intervention": signal.weight > 0.5,
                    "expected_circuit_breaker": False,
                    "expected_signal": signal.name,
                    "status": "draft",
                }
            )
            counts["risk_signal"] += 1

    # Generate uncertainty type cases
    for utype in spec.uncertainty_types:
        if utype.name == "epistemic":
            test_cases.append(
                {
                    "name": f"uncertainty_{utype.name}_knowledge_gap",
                    "category": "uncertainty_type",
                    "query": "What is the exact mechanism behind quantum decoherence in biological systems?",
                    "expected_intervention": True,
                    "expected_uncertainty_type": utype.name,
                    "status": "draft",
                }
            )
        elif utype.name == "contradictory":
            test_cases.append(
                {
                    "name": f"uncertainty_{utype.name}_conflict",
                    "category": "uncertainty_type",
                    "query": "Is coffee good or bad for cardiovascular health?",
                    "expected_intervention": True,
                    "expected_uncertainty_type": utype.name,
                    "status": "draft",
                }
            )
        elif utype.name == "open_ended":
            test_cases.append(
                {
                    "name": f"uncertainty_{utype.name}_multiple_views",
                    "category": "uncertainty_type",
                    "query": "What is the best programming language for building web applications?",
                    "expected_intervention": True,
                    "expected_uncertainty_type": utype.name,
                    "status": "draft",
                }
            )
        counts["uncertainty_type"] += 1

    # Generate safe passthrough cases
    safe_cases = [
        {
            "name": "safe_general_knowledge",
            "category": "safe_passthrough",
            "query": "What color is the sky?",
            "expected_intervention": False,
            "expected_circuit_breaker": False,
            "status": "draft",
        },
        {
            "name": "safe_math",
            "category": "safe_passthrough",
            "query": "What is 2 + 2?",
            "expected_intervention": False,
            "expected_circuit_breaker": False,
            "status": "draft",
        },
        {
            "name": "safe_greeting",
            "category": "safe_passthrough",
            "query": "Hello, how are you today?",
            "expected_intervention": False,
            "expected_circuit_breaker": False,
            "status": "draft",
        },
    ]
    test_cases.extend(safe_cases)

    # Write the test suite
    suite = {
        "metadata": {
            "spec_name": spec.metadata.name,
            "spec_version": spec.metadata.version,
            "generated_by": "brix generate-tests",
            "status": "draft",
            "total_cases": len(test_cases),
        },
        "test_cases": test_cases,
    }

    output_file = output_dir / f"test_suite_{spec.metadata.name}_{spec.metadata.version}.yaml"
    output_file.write_text(
        yaml.dump(suite, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )

    # Summary
    table = Table(title="Generated Test Suite Summary")
    table.add_column("Category", style="bold")
    table.add_column("Count")
    table.add_row("Circuit Breaker (positive)", str(counts["circuit_breaker_positive"]))
    table.add_row("Circuit Breaker (negative)", str(counts["circuit_breaker_negative"]))
    table.add_row("Risk Signal", str(counts["risk_signal"]))
    table.add_row("Uncertainty Type", str(counts["uncertainty_type"]))
    table.add_row("Safe Passthrough", str(len(safe_cases)))
    table.add_row("Total", str(len(test_cases)))
    console.print(table)

    console.print(f"\n[green]Generated {len(test_cases)} test cases[/green]")
    console.print(f"Output: {output_file}")
    console.print("\n[dim]All generated tests have status: draft — review before use[/dim]")

    raise typer.Exit(code=0)


def _slugify(text: str) -> str:
    """Convert text to a slug suitable for test case names."""
    return text.lower().replace(" ", "_").replace("'", "").replace('"', "")[:40]
