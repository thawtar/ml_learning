"""Interactive session management for the OpenFOAM LLM Wrapper."""

import logging
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from openfoam_llm_wrapper.core.context_manager import ConversationContext
from openfoam_llm_wrapper.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
console = Console()


class InteractiveSession:
    """Manages an interactive chat session with the user."""

    def __init__(self):
        """Initialize the interactive session."""
        self.orchestrator = Orchestrator()
        self.context = ConversationContext()
        self.running = False

    def run_interactive(self) -> None:
        """Run the interactive chat loop."""
        self.running = True
        self._print_welcome()

        try:
            while self.running:
                user_input = Prompt.ask(
                    "[bold cyan]You[/bold cyan]",
                    console=console,
                ).strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    self._print_goodbye()
                    self.running = False
                    break

                if user_input.lower() == "help":
                    self._print_help()
                    continue

                if user_input.lower() == "clear":
                    self.context.clear_history()
                    console.print("[green]Conversation history cleared[/green]")
                    continue

                self._process_user_input(user_input)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted[/yellow]")
            self.running = False
        except Exception as e:
            console.print(f"[red]Unexpected error: {str(e)}[/red]")
            logger.exception("Error in interactive session")

    def _process_user_input(self, user_input: str) -> None:
        """
        Process user input and generate response.

        Routes to either regular conversation or interactive workflow
        based on detected intent.

        Args:
            user_input: The user's input text
        """
        self.context.add_message("user", user_input)

        try:
            # Check if user wants interactive case generation
            if self._is_case_generation_intent(user_input):
                self._start_interactive_workflow()
                return

            # Otherwise, treat as regular conversation
            response = self._get_response(user_input)

            # Add to context and display
            self.context.add_message("assistant", response)
            console.print(
                Panel(
                    response,
                    title="[bold cyan]Assistant[/bold cyan]",
                    style="cyan",
                )
            )

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            logger.exception("Error processing user input")

    def _get_response(self, user_input: str) -> str:
        """
        Get response from the orchestrator based on user input.

        Args:
            user_input: The user's input text

        Returns:
            Response string
        """
        # This is a placeholder - will be enhanced with intent-based routing
        # For now, treat everything as a general question
        response = self.orchestrator.llm_client.answer_question(user_input)
        return response

    def _is_case_generation_intent(self, text: str) -> bool:
        """
        Detect if user wants to start case generation.

        Args:
            text: User input text

        Returns:
            True if case generation is intended
        """
        keywords = [
            "generate",
            "create case",
            "setup simulation",
            "interactive",
            "build case",
            "new case",
        ]

        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def _start_interactive_workflow(self) -> None:
        """
        Start the interactive case generation workflow.

        This routes to the InteractiveWorkflowManager which handles
        the guided step-by-step case generation process.
        """
        from openfoam_llm_wrapper.core.interactive_workflow import (
            InteractiveWorkflowManager,
        )

        try:
            console.print(
                Panel(
                    "Starting interactive case generation workflow...",
                    style="cyan",
                )
            )

            # Create and run workflow manager
            workflow = InteractiveWorkflowManager(
                context=self.context, llm_client=self.orchestrator.llm_client
            )

            workflow.start_workflow()

            # After workflow completes, display summary
            if not workflow.is_workflow_complete():
                case_info = workflow.get_case_info()

                # Display final summary
                from openfoam_llm_wrapper.core.summary_formatter import (
                    display_category_status,
                )

                console.print("")
                display_category_status(case_info)

                # Offer to generate files
                console.print(
                    Panel(
                        "Type 'yes' to generate case files, or 'no' to cancel",
                        style="cyan",
                    )
                )

                confirm = Prompt.ask("[bold cyan]Generate files?[/bold cyan]")

                if confirm.lower() in ["yes", "y"]:
                    self._generate_from_workflow_data(case_info)
                else:
                    console.print("[yellow]Case generation cancelled[/yellow]")

        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Workflow error: {str(e)}[/bold red]",
                    style="red",
                )
            )
            logger.exception("Error in interactive workflow")

    def _generate_from_workflow_data(self, case_info: dict) -> None:
        """
        Generate OpenFOAM case from collected workflow data.

        Args:
            case_info: The collected case information dictionary
        """
        try:
            console.print(
                Panel(
                    "Generating OpenFOAM case files...", style="cyan"
                )
            )

            # Use orchestrator to generate case from structured data
            result = self.orchestrator.generate_case_from_interactive(
                case_info=case_info,
                output_dir=".",
                case_name=case_info.get("case_name", "openfoam_case"),
            )

            # Display success message
            case_path = result.get("case_path", "unknown")
            validation = result.get("validation", {})

            success_text = f"""OpenFOAM case generated successfully!

Location: {case_path}
Validation: {'✓ Passed' if validation.get('is_valid') else '⚠ Warnings detected'}

You can now:
1. Review the generated files
2. Run blockMesh (if needed): blockMesh -case {case_path}
3. Run your solver: cd {case_path} && <solver_name>
"""

            console.print(Panel(success_text, style="green", title="Success"))

            # Add to conversation history
            self.context.add_message("assistant", success_text)

        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Generation error: {str(e)}[/bold red]",
                    style="red",
                )
            )
            logger.exception("Error generating case")

    def _print_welcome(self) -> None:
        """Print welcome message."""
        welcome_text = """Welcome to the [bold cyan]OpenFOAM LLM Assistant[/bold cyan]!

I can help you:
  • [yellow]Generate[/yellow] OpenFOAM cases from descriptions
  • [yellow]Explain[/yellow] error messages
  • [yellow]Recommend[/yellow] appropriate solvers
  • [yellow]Answer[/yellow] questions about OpenFOAM

Type [bold]help[/bold] for more information, or just start chatting!
"""
        console.print(Panel(welcome_text, style="green", title="Welcome"))

    def _print_goodbye(self) -> None:
        """Print goodbye message."""
        console.print("[cyan]Thanks for using OpenFOAM LLM Assistant. Goodbye![/cyan]")

    def _print_help(self) -> None:
        """Print help information."""
        help_text = """
[bold]Commands:[/bold]
  help    - Show this help message
  clear   - Clear conversation history
  exit    - Exit the session

[bold]Example Queries:[/bold]
  "Generate a steady-state incompressible flow case for a pipe"
  "What does this error mean: Foam::error::printStack"
  "Recommend a solver for transient heat transfer"
  "How do I set up turbulence inlet conditions?"
"""
        console.print(Panel(help_text, title="Help"))
