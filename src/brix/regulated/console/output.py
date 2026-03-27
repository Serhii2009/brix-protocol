"""Console output --optional visual feedback for developers.

Enabled automatically when stdout is a TTY. Override with:
  BRIX_CONSOLE=1  forces on (even in pipes)
  BRIX_CONSOLE=0  forces off (even in terminal)
  BRIX_VERBOSE=1  adds signal and retrieval detail
"""

from __future__ import annotations

import os
import sys

from brix.regulated.core.result import StructuredResult
from brix.regulated.output.result import OutputResult


def _is_enabled() -> bool:
    """Check whether console output is enabled."""
    env = os.environ.get("BRIX_CONSOLE")
    if env == "1":
        return True
    if env == "0":
        return False
    return sys.stdout.isatty()


def _is_verbose() -> bool:
    """Check whether verbose mode is enabled."""
    return os.environ.get("BRIX_VERBOSE") == "1"


def print_result(
    result: StructuredResult,
    *,
    output_result: OutputResult | None = None,
) -> None:
    """Print a rich Panel summary for a BRIX decision.

    This function never raises -- console failures are logged to stderr.

    Args:
        result: The StructuredResult to display.
        output_result: Optional OutputResult from output guard.
    """
    try:
        if not _is_enabled():
            return

        from rich.console import Console
        from rich.panel import Panel

        console = Console(
            stderr=True,
            highlight=False,
            force_terminal=True,
            width=60,
        )
        verbose = _is_verbose()

        # Determine status
        if result.circuit_breaker_hit:
            icon = "X"
            label = f"BLOCKED -- {result.circuit_breaker_name}"
            border = "red"
        elif result.intervention_necessary:
            icon = "!"
            label = "ELEVATED -- retrieval needed"
            border = "yellow"
        else:
            icon = "OK"
            label = "SAFE -- passed through"
            border = "green"

        # Build panel body
        lines: list[str] = []
        lines.append(
            f"Risk: {result.risk_score:.2f}  |  "
            f"Balance: {result.balance_index:.3f}  |  "
            f"{result.latency_ms:.0f}ms"
        )

        # Output guard blocked line
        if output_result is not None and output_result.output_blocked:
            block_sig = output_result.output_block_signal or "unknown"
            lines.append(f"[red]Response blocked -- {block_sig}[/red]")

        # Verbose detail
        if verbose:
            if result.signals_triggered:
                lines.append(f"Signals: {', '.join(result.signals_triggered)}")
            if result.retrieval_executed:
                src_count = len(result.retrieval_sources)
                lines.append(f"Retrieval: executed -- {src_count} source(s)")
            elif result.retrieval_failed:
                lines.append("Retrieval: [red]failed[/red]")
            lines.append(f"Decision: {result.decision_id}")

        body = "\n".join(lines)
        title = f" {icon}  {label} "
        panel = Panel(body, title=title, border_style=border, expand=False)
        console.print(panel)

    except Exception as exc:
        print(f"BRIX console warning: {exc}", file=sys.stderr)
