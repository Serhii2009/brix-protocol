"""BRIX CLI entry point — the `brix` command."""

import typer

from brix.regulated.cli.explain import explain_cmd
from brix.regulated.cli.generate_tests import generate_tests_cmd
from brix.regulated.cli.lint import lint_cmd
from brix.regulated.cli.test_cmd import test_cmd

app = typer.Typer(
    name="brix",
    help="BRIX — Runtime Reliability Infrastructure for LLM Pipelines",
    no_args_is_help=True,
)

app.command(name="lint", help="Validate and analyze an uncertainty.yaml specification")(lint_cmd)
app.command(name="test", help="Run a test suite against a specification")(test_cmd)
app.command(name="explain", help="Reconstruct a decision trace from structured result logs")(explain_cmd)
app.command(name="generate-tests", help="Generate a draft test suite from a specification")(generate_tests_cmd)


if __name__ == "__main__":
    app()
