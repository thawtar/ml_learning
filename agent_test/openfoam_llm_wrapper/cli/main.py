"""Main CLI entry point for OpenFOAM LLM Wrapper."""

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__)
def cli():
    """OpenFOAM LLM Wrapper - AI-powered assistant for CFD engineers."""
    pass


@cli.command()
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Start in interactive mode",
)
def chat(interactive: bool):
    """Start an interactive chat session with the OpenFOAM assistant."""
    from openfoam_llm_wrapper.core.session import InteractiveSession

    session = InteractiveSession()
    if interactive:
        session.run_interactive()
    else:
        console.print(
            Panel(
                "Use [bold]-i[/bold] or [bold]--interactive[/bold] flag to start interactive mode",
                title="Chat Mode",
            )
        )


@cli.command()
@click.argument("description", required=False)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    help="Output directory for generated case (default: current directory)",
)
@click.option(
    "--case-name",
    "-n",
    default="openfoam_case",
    help="Name of the case directory to create",
)
def generate(description: str, output_dir: str, case_name: str):
    """Generate an OpenFOAM case from a description."""
    if not description:
        console.print(
            "[bold red]Error:[/bold red] Please provide a case description as an argument"
        )
        return

    from openfoam_llm_wrapper.core.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        result = orchestrator.generate_case(description, output_dir, case_name)
        console.print(
            Panel(
                f"Case generated successfully at: [bold]{result['case_path']}[/bold]",
                title="Success",
                style="green",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                title="Generation Failed",
                style="red",
            )
        )


@cli.command()
@click.argument("error_message")
def explain(error_message: str):
    """Explain an OpenFOAM error message."""
    from openfoam_llm_wrapper.core.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        explanation = orchestrator.explain_error(error_message)
        console.print(
            Panel(
                explanation,
                title="Error Explanation",
                style="yellow",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                title="Explanation Failed",
                style="red",
            )
        )


@cli.command()
@click.argument("physics_description", required=False)
def recommend(physics_description: str):
    """Get solver recommendations based on your simulation physics."""
    if not physics_description:
        console.print(
            "[bold red]Error:[/bold red] Please describe your simulation physics"
        )
        return

    from openfoam_llm_wrapper.core.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        recommendation = orchestrator.recommend_solver(physics_description)
        console.print(
            Panel(
                recommendation,
                title="Solver Recommendation",
                style="cyan",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                title="Recommendation Failed",
                style="red",
            )
        )


if __name__ == "__main__":
    cli()
