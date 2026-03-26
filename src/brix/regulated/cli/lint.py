"""brix lint — validate schema, detect conflicts, estimate Balance Index.

Exit codes:
  0 — clean (no warnings, no errors)
  1 — warnings detected
  2 — errors detected
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from brix.regulated.core.exceptions import SpecValidationError
from brix.regulated.spec.loader import load_spec
from brix.regulated.spec.models import SpecModel

console = Console()


def lint_cmd(
    spec_file: str = typer.Argument(help="Path to the uncertainty.yaml specification file"),
) -> None:
    """Validate and analyze a BRIX specification file."""
    path = Path(spec_file)
    errors: list[str] = []
    warnings: list[str] = []

    # Step 1: Load and validate schema
    console.print(f"\n[bold]BRIX Lint[/bold] — {path.name}\n")

    try:
        spec = load_spec(path)
    except SpecValidationError as exc:
        console.print(f"[red]ERROR[/red] Schema validation failed:\n{exc}")
        raise typer.Exit(code=2)

    console.print("[green]OK[/green] Schema validation passed")

    # Step 2: Detect conflicting signals
    conflicts = _detect_conflicts(spec)
    for conflict in conflicts:
        errors.append(conflict)
        console.print(f"[red]ERROR[/red] {conflict}")

    # Step 2b: Detect conflicting output signals
    output_conflicts = _detect_output_conflicts(spec)
    for conflict in output_conflicts:
        errors.append(conflict)
        console.print(f"[red]ERROR[/red] {conflict}")

    # Step 3: Detect unreachable rules
    unreachable = _detect_unreachable(spec)
    for rule in unreachable:
        warnings.append(rule)
        console.print(f"[yellow]WARN[/yellow] {rule}")

    # Step 3b: Detect unreachable output signals
    output_unreachable = _detect_output_unreachable(spec)
    for rule in output_unreachable:
        warnings.append(rule)
        console.print(f"[yellow]WARN[/yellow] {rule}")

    # Step 4: Estimate utility impact
    utility_impact = _estimate_utility_impact(spec)
    if utility_impact > 0.50:
        warnings.append(
            f"High utility impact: estimated {utility_impact:.0%} of queries may trigger elevated handling"
        )
        console.print(
            f"[yellow]WARN[/yellow] High utility impact: {utility_impact:.0%} of queries may trigger intervention"
        )

    # Step 5: Estimate Balance Index
    estimated_balance = _estimate_balance_index(spec, utility_impact)

    # Count output signal types
    output_total = len(spec.output_signals)
    output_block = sum(1 for s in spec.output_signals if s.signal_type == "block")

    # Summary table
    console.print()
    table = Table(title="Lint Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Specification", f"{spec.metadata.name} v{spec.metadata.version}")
    table.add_row("Domain", spec.metadata.domain)
    table.add_row("Circuit Breakers", str(len(spec.circuit_breakers)))
    table.add_row("Risk Signals", str(len(spec.risk_signals)))
    table.add_row("Output Signals", str(output_total))
    table.add_row("Output Block Signals", str(output_block))
    table.add_row("Total Patterns", str(_count_patterns(spec)))
    table.add_row("Estimated Utility Impact", f"{utility_impact:.1%}")
    table.add_row("Estimated Balance Index", f"{estimated_balance:.3f}")
    table.add_row("Errors", str(len(errors)))
    table.add_row("Warnings", str(len(warnings)))
    console.print(table)
    console.print()

    if errors:
        console.print(f"[red]Lint failed with {len(errors)} error(s)[/red]")
        raise typer.Exit(code=2)
    elif warnings:
        console.print(f"[yellow]Lint passed with {len(warnings)} warning(s)[/yellow]")
        raise typer.Exit(code=1)
    else:
        console.print("[green]Lint passed — specification is clean[/green]")
        raise typer.Exit(code=0)


def _detect_conflicts(spec: SpecModel) -> list[str]:
    """Detect conflicting signals where patterns overlap between CB and risk signals."""
    conflicts: list[str] = []
    cb_patterns: dict[str, str] = {}

    for cb in spec.circuit_breakers:
        for pattern in cb.patterns:
            key = pattern.lower()
            cb_patterns[key] = cb.name

    for signal in spec.risk_signals:
        for pattern in signal.patterns:
            key = pattern.lower()
            if key in cb_patterns:
                conflicts.append(
                    f"Pattern '{pattern}' appears in both circuit breaker "
                    f"'{cb_patterns[key]}' and risk signal '{signal.name}'"
                )

    return conflicts


def _detect_output_conflicts(spec: SpecModel) -> list[str]:
    """Detect conflicting patterns between output signals and CB/risk signals."""
    conflicts: list[str] = []
    cb_patterns: dict[str, str] = {}

    for cb in spec.circuit_breakers:
        for pattern in cb.patterns:
            cb_patterns[pattern.lower()] = cb.name

    for signal in spec.output_signals:
        for pattern in signal.patterns:
            key = pattern.lower()
            if key in cb_patterns:
                conflicts.append(
                    f"Output signal pattern '{pattern}' conflicts with circuit breaker "
                    f"'{cb_patterns[key]}'"
                )

    return conflicts


def _detect_unreachable(spec: SpecModel) -> list[str]:
    """Detect rules where exclude_context would eliminate all possible matches."""
    unreachable: list[str] = []

    for cb in spec.circuit_breakers:
        if not cb.exclude_context:
            continue
        # A CB is "unreachable" if every pattern is a substring of some exclusion
        all_excluded = True
        for pattern in cb.patterns:
            p_lower = pattern.lower()
            if not any(p_lower in exc.lower() for exc in cb.exclude_context):
                all_excluded = False
                break
        if all_excluded:
            unreachable.append(
                f"Circuit breaker '{cb.name}' may be unreachable: all patterns "
                f"are substrings of exclude_context entries"
            )

    for signal in spec.risk_signals:
        if not signal.exclude_context:
            continue
        all_excluded = True
        for pattern in signal.patterns:
            p_lower = pattern.lower()
            if not any(p_lower in exc.lower() for exc in signal.exclude_context):
                all_excluded = False
                break
        if all_excluded:
            unreachable.append(
                f"Risk signal '{signal.name}' may be unreachable: all patterns "
                f"are substrings of exclude_context entries"
            )

    return unreachable


def _detect_output_unreachable(spec: SpecModel) -> list[str]:
    """Detect output signals where exclude_context eliminates all matches."""
    unreachable: list[str] = []

    for signal in spec.output_signals:
        if not signal.exclude_context:
            continue
        all_excluded = True
        for pattern in signal.patterns:
            p_lower = pattern.lower()
            if not any(p_lower in exc.lower() for exc in signal.exclude_context):
                all_excluded = False
                break
        if all_excluded:
            unreachable.append(
                f"Output signal '{signal.name}' may be unreachable: all patterns "
                f"are substrings of exclude_context entries"
            )

    return unreachable


def _estimate_utility_impact(spec: SpecModel) -> float:
    """Estimate the proportion of typical queries that would trigger elevated handling.

    Simple heuristic based on the number and breadth of patterns.
    """
    total_patterns = _count_patterns(spec)
    total_exclusions = sum(len(cb.exclude_context) for cb in spec.circuit_breakers) + sum(
        len(s.exclude_context) for s in spec.risk_signals
    )

    # More patterns = more triggers; more exclusions = fewer false positives
    raw_impact = min(total_patterns * 0.01, 1.0)
    exclusion_factor = max(0.3, 1.0 - total_exclusions * 0.02)
    return min(raw_impact * exclusion_factor, 1.0)


def _estimate_balance_index(spec: SpecModel, utility_impact: float) -> float:
    """Estimate the Balance Index based on spec characteristics.

    This is a static analysis estimate, not based on actual test results.
    """
    # Heuristic: specs with good exclusion coverage tend to have higher balance
    total_patterns = _count_patterns(spec)
    if total_patterns == 0:
        return 0.0

    total_exclusions = sum(len(cb.exclude_context) for cb in spec.circuit_breakers) + sum(
        len(s.exclude_context) for s in spec.risk_signals
    )

    # Estimated reliability: more patterns = better coverage
    est_reliability = min(0.95, 0.70 + total_patterns * 0.005)

    # Estimated utility: lower impact = higher utility
    est_utility = max(0.50, 1.0 - utility_impact)

    # Balance Index = harmonic mean
    if est_reliability + est_utility == 0:
        return 0.0
    return 2.0 * est_reliability * est_utility / (est_reliability + est_utility)


def _count_patterns(spec: SpecModel) -> int:
    """Count total patterns across all circuit breakers, risk signals, and output signals."""
    count = sum(len(cb.patterns) for cb in spec.circuit_breakers)
    count += sum(len(s.patterns) for s in spec.risk_signals)
    count += sum(len(s.patterns) for s in spec.output_signals)
    return count
