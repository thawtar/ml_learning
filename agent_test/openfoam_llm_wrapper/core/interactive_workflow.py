"""Interactive workflow for case generation with step-by-step questioning."""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openfoam_llm_wrapper.core.context_manager import ConversationContext

logger = logging.getLogger(__name__)
console = Console()


class WorkflowState(Enum):
    """Enumeration of workflow states."""

    INITIALIZATION = "initialization"
    GATHERING_INFO = "gathering_info"
    REVIEWING_SUMMARY = "reviewing_summary"
    CONFIRMED = "confirmed"
    GENERATING = "generating"
    COMPLETE = "complete"


class InformationCategory(Enum):
    """Enumeration of information categories to gather."""

    PHYSICS = "physics"
    GEOMETRY = "geometry"
    BOUNDARY_CONDITIONS = "boundary_conditions"
    FLUID_PROPERTIES = "fluid_properties"
    SOLVER = "solver"
    MESH = "mesh"
    SIMULATION_GOALS = "simulation_goals"
    ADVANCED = "advanced"


@dataclass
class QuestionRecord:
    """Record of a question and answer."""

    topic: str
    question: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)


class InformationCollector:
    """Manages case_info structure and data collection."""

    # Required fields for each category
    REQUIRED_FIELDS = {
        InformationCategory.PHYSICS: ["flow_type", "time_type"],
        InformationCategory.GEOMETRY: ["description"],
        InformationCategory.BOUNDARY_CONDITIONS: ["patches"],
        InformationCategory.SOLVER: ["solver"],
    }

    # Optional categories
    OPTIONAL_CATEGORIES = [
        InformationCategory.FLUID_PROPERTIES,
        InformationCategory.MESH,
        InformationCategory.SIMULATION_GOALS,
        InformationCategory.ADVANCED,
    ]

    def __init__(self):
        """Initialize the information collector."""
        self.case_info = self._create_empty_structure()

    def _create_empty_structure(self) -> Dict[str, Any]:
        """Create the empty case_info structure."""
        return {
            "workflow_state": WorkflowState.INITIALIZATION.value,
            "collection_status": {cat.value: False for cat in InformationCategory},
            "collected_data": {
                InformationCategory.PHYSICS.value: {},
                InformationCategory.GEOMETRY.value: {},
                InformationCategory.BOUNDARY_CONDITIONS.value: {},
                InformationCategory.FLUID_PROPERTIES.value: {},
                InformationCategory.SOLVER.value: {},
                InformationCategory.MESH.value: {},
                InformationCategory.SIMULATION_GOALS.value: {},
                InformationCategory.ADVANCED.value: {},
            },
            "question_history": [],
            "user_edits": [],
        }

    def initialize(self) -> None:
        """Initialize case_info structure."""
        self.case_info = self._create_empty_structure()
        logger.info("Initialized case_info structure")

    def update_field(self, category: str, key: str, value: Any) -> None:
        """
        Update a field in the collected data.

        Args:
            category: Category name (e.g., "physics")
            key: Field name (e.g., "flow_type")
            value: Value to set
        """
        if category not in self.case_info["collected_data"]:
            logger.warning(f"Unknown category: {category}")
            return

        self.case_info["collected_data"][category][key] = value
        logger.debug(f"Updated {category}.{key} = {value}")

    def update_nested_field(self, path: str, value: Any) -> None:
        """
        Update a nested field using dot notation.

        Args:
            path: Dot-notation path (e.g., "physics.flow_type")
            value: Value to set
        """
        parts = path.split(".")
        if len(parts) == 2:
            category, key = parts
            self.update_field(category, key, value)
        else:
            logger.warning(f"Invalid path format: {path}")

    def get_field(self, category: str, key: str, default: Any = None) -> Any:
        """Get a field from collected data."""
        if category not in self.case_info["collected_data"]:
            return default

        return self.case_info["collected_data"][category].get(key, default)

    def mark_category_complete(self, category: str) -> None:
        """Mark a category as complete."""
        if category in self.case_info["collection_status"]:
            self.case_info["collection_status"][category] = True
            logger.debug(f"Marked {category} as complete")

    def mark_category_incomplete(self, category: str) -> None:
        """Mark a category as incomplete."""
        if category in self.case_info["collection_status"]:
            self.case_info["collection_status"][category] = False
            logger.debug(f"Marked {category} as incomplete")

    def get_completion_status(self) -> Dict[str, bool]:
        """Get completion status of all categories."""
        return self.case_info["collection_status"].copy()

    def get_missing_required(self) -> List[str]:
        """Get list of missing required information."""
        missing = []

        for category, required_fields in self.REQUIRED_FIELDS.items():
            category_value = category.value
            category_data = self.case_info["collected_data"][category_value]

            for field_name in required_fields:
                if field_name not in category_data or not category_data[field_name]:
                    missing.append(f"{category_value}.{field_name}")

        return missing

    def add_question_to_history(
        self, topic: str, question: str, answer: str
    ) -> None:
        """
        Add a question-answer pair to history.

        Args:
            topic: Topic/category of the question
            question: The question asked
            answer: The user's answer
        """
        record = QuestionRecord(topic=topic, question=question, answer=answer)
        self.case_info["question_history"].append(
            {
                "topic": record.topic,
                "question": record.question,
                "answer": record.answer,
                "timestamp": record.timestamp.isoformat(),
            }
        )
        logger.debug(f"Added question to history: {topic}")

    def get_question_history(self) -> List[Dict[str, Any]]:
        """Get the question history."""
        return self.case_info["question_history"].copy()

    def get_questions_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all questions for a specific topic."""
        return [q for q in self.case_info["question_history"] if q["topic"] == topic]

    def is_complete(self) -> bool:
        """Check if all required information is collected."""
        missing = self.get_missing_required()
        return len(missing) == 0

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate the collected information.

        Returns:
            Dictionary with 'errors' and 'warnings' keys
        """
        errors = []
        warnings = []

        # Check required fields
        for category, required_fields in self.REQUIRED_FIELDS.items():
            category_value = category.value
            category_data = self.case_info["collected_data"][category_value]

            for field_name in required_fields:
                if field_name not in category_data or not category_data[field_name]:
                    errors.append(f"Missing required: {category_value}.{field_name}")

        # Check if physics is reasonable (example)
        physics_data = self.case_info["collected_data"].get("physics", {})
        if "flow_type" in physics_data and physics_data["flow_type"] not in [
            "incompressible",
            "compressible",
        ]:
            warnings.append(
                f"Unusual flow_type value: {physics_data.get('flow_type')}"
            )

        return {"errors": errors, "warnings": warnings}


class CommandHandler:
    """Handles user commands during the workflow."""

    VALID_COMMANDS = {
        "skip": "Skip current question/topic",
        "back": "Go back and edit a previous answer",
        "summary": "Show current collected information",
        "edit": "Edit a specific field",
        "done": "Check if ready to generate",
        "generate": "Force proceed to summary",
        "help": "Show available commands",
        "quit": "Exit the workflow",
        "cancel": "Exit the workflow",
    }

    def __init__(self, collector: InformationCollector):
        """
        Initialize the command handler.

        Args:
            collector: InformationCollector instance
        """
        self.collector = collector

    def parse_command(self, user_input: str) -> Optional[tuple[str, Optional[str]]]:
        """
        Parse user input to detect commands.

        Returns:
            Tuple of (command_name, argument) or None if not a command
        """
        user_input = user_input.strip().lower()

        # Check for direct commands
        for cmd in self.VALID_COMMANDS.keys():
            if user_input == cmd:
                return (cmd, None)

            # Check for commands with arguments (e.g., "edit physics")
            if user_input.startswith(f"{cmd} "):
                argument = user_input[len(cmd) :].strip()
                return (cmd, argument if argument else None)

        return None

    def is_command(self, user_input: str) -> bool:
        """Check if input is a command."""
        return self.parse_command(user_input) is not None

    def execute_command(
        self, command: str, argument: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a command.

        Args:
            command: Command name
            argument: Optional argument

        Returns:
            Dictionary with result information
        """
        logger.debug(f"Executing command: {command} {argument or ''}")

        if command == "skip":
            return {"action": "skip", "message": "Skipped to next topic"}

        elif command == "back":
            return self._handle_back()

        elif command == "summary":
            return {"action": "summary", "message": "Showing summary"}

        elif command == "edit":
            if not argument:
                return {
                    "action": "error",
                    "message": "Usage: edit <field> (e.g., edit physics.flow_type)",
                }
            return {"action": "edit", "field": argument}

        elif command == "done":
            is_complete = self.collector.is_complete()
            missing = self.collector.get_missing_required()
            return {
                "action": "done",
                "is_complete": is_complete,
                "missing": missing,
            }

        elif command == "generate":
            return {"action": "generate", "message": "Proceeding to summary"}

        elif command == "help":
            return self._handle_help()

        elif command in ["quit", "cancel"]:
            return {"action": "exit", "message": "Exiting workflow"}

        else:
            return {"action": "error", "message": f"Unknown command: {command}"}

    def _handle_back(self) -> Dict[str, Any]:
        """Handle the 'back' command."""
        history = self.collector.get_question_history()

        if not history:
            return {
                "action": "back",
                "found_history": False,
                "message": "No previous questions to edit",
            }

        return {
            "action": "back",
            "found_history": True,
            "history": history,
            "message": f"Found {len(history)} previous questions",
        }

    def _handle_help(self) -> Dict[str, Any]:
        """Handle the 'help' command."""
        return {
            "action": "help",
            "commands": self.VALID_COMMANDS,
            "message": "Available commands",
        }


class InteractiveWorkflowManager:
    """Manages the interactive case generation workflow."""

    def __init__(self, context: ConversationContext, llm_client: Any):
        """
        Initialize the workflow manager.

        Args:
            context: ConversationContext instance for message history
            llm_client: LLMClient instance for LLM interactions
        """
        self.context = context
        self.llm_client = llm_client
        self.collector = InformationCollector()
        self.command_handler = CommandHandler(self.collector)
        self.state = WorkflowState.INITIALIZATION
        self.question_count = 0
        self.max_questions = 20

        logger.info("Initialized InteractiveWorkflowManager")

    def initialize_workflow(self) -> None:
        """Initialize the workflow and display welcome message."""
        self.collector.initialize()
        self.state = WorkflowState.INITIALIZATION
        self._display_welcome()
        self.state = WorkflowState.GATHERING_INFO

    def start_workflow(self) -> None:
        """Start the interactive workflow (main entry point)."""
        try:
            self.initialize_workflow()
            self._run_gathering_loop()
        except KeyboardInterrupt:
            console.print("\n[yellow]Workflow interrupted[/yellow]")
            logger.info("Workflow interrupted by user")
        except Exception as e:
            console.print(f"[red]Error in workflow: {str(e)}[/red]")
            logger.exception("Error in workflow")

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """Interactive Case Generation

I'll guide you through setting up your OpenFOAM case by asking detailed technical questions.

You can use these commands at any time:
  • [bold]skip[/bold]     - Skip to the next topic
  • [bold]back[/bold]     - Go back and edit previous answers
  • [bold]summary[/bold]  - Show what we've collected so far
  • [bold]edit <field>[/bold] - Edit a specific field
  • [bold]done[/bold]     - Check if we're ready to generate
  • [bold]help[/bold]     - Show all commands
  • [bold]quit[/bold]     - Exit the workflow
"""
        console.print(Panel(welcome_text, title="Welcome", style="green"))

    def _run_gathering_loop(self) -> None:
        """Run the main information gathering loop."""
        from rich.prompt import Prompt

        max_attempts = 100  # Safety limit
        attempts = 0

        while (
            self.state == WorkflowState.GATHERING_INFO
            and attempts < max_attempts
        ):
            attempts += 1

            # Get next question from LLM
            question = self._get_next_question()
            if not question:
                logger.warning("Failed to get next question from LLM")
                console.print(
                    "[red]Error getting next question. Please try again.[/red]"
                )
                continue

            # Display question
            self._display_question(question)

            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()

            if not user_input:
                console.print("[yellow]Please provide an answer[/yellow]")
                continue

            # Handle commands
            if self.command_handler.is_command(user_input):
                self._handle_command(user_input)
                continue

            # Process regular answer
            self._process_answer(question, user_input)

            # Check if we should move to summary
            if self._should_transition_to_summary():
                self.transition_to_summary()
                break

        if attempts >= max_attempts:
            logger.warning("Reached max attempts in gathering loop")
            console.print(
                "[yellow]Reached maximum questions. Moving to summary.[/yellow]"
            )
            self.transition_to_summary()

    def _get_next_question(self) -> Optional[str]:
        """
        Get the next question from the LLM.

        Uses multi-turn conversation with LLM to generate intelligent,
        context-aware questions based on what's already been collected.

        Returns:
            The question text or None if error/completion
        """
        self.question_count += 1

        # Safety limit
        if self.question_count > self.max_questions:
            logger.warning("Exceeded maximum questions")
            return None

        try:
            # Build knowledge base context
            from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

            prompt_builder = PromptBuilder()
            knowledge_context = prompt_builder.build_knowledge_base_context()

            # Get conversation history from context
            conversation_history = self.context.get_conversation_history()

            # Get collection status and data
            collection_status = self.collector.get_completion_status()
            collected_data = self.collector.case_info["collected_data"]
            question_history = self.collector.get_question_history()

            # Call LLM to get next question
            response = self.llm_client.interactive_case_questioning(
                conversation_history=conversation_history,
                collection_status=collection_status,
                collected_data=collected_data,
                question_history=question_history,
                knowledge_context=knowledge_context,
            )

            # Add LLM response to conversation history
            self.context.add_message("assistant", response)

            # Check if LLM signals completion
            if self._is_completion_signal(response):
                self.state = WorkflowState.REVIEWING_SUMMARY
                return None

            # Extract just the question part (remove any metadata/formatting)
            question = self._extract_question_from_response(response)

            if not question:
                logger.warning("Could not extract question from LLM response")
                return None

            logger.debug(f"Got question {self.question_count}: {question[:100]}")
            return question

        except Exception as e:
            logger.error(f"Error getting next question from LLM: {str(e)}")
            return None

    def _is_completion_signal(self, response: str) -> bool:
        """
        Check if LLM response contains completion signal.

        Args:
            response: The LLM response text

        Returns:
            True if response signals completion
        """
        completion_phrases = [
            "READY_TO_GENERATE",
            "ready to generate",
            "sufficient information",
            "enough information",
        ]

        response_lower = response.lower()
        return any(phrase.lower() in response_lower for phrase in completion_phrases)

    def _extract_question_from_response(self, response: str) -> Optional[str]:
        """
        Extract the actual question from LLM response.

        The LLM might return formatted text, so this extracts the core question.

        Args:
            response: The LLM response text

        Returns:
            The extracted question or None
        """
        # Remove common prefixes/formatting
        lines = response.strip().split("\n")

        # Find the first substantial line that looks like a question
        for line in lines:
            line = line.strip()

            # Skip empty lines and metadata
            if not line or line.startswith("[") or line.startswith("#"):
                continue

            # Check if it looks like a question
            if "?" in line or line[0].isupper():
                return line

        # If no question found, return the whole response
        if response and len(response) > 20:
            return response[:200]  # Return first 200 chars

        return None

    def _display_question(self, question: str) -> None:
        """Display a question to the user."""
        status = (
            f"Question {self.question_count}/~{self.max_questions}"
        )
        console.print(
            Panel(
                question,
                title=status,
                style="cyan",
                expand=False,
            )
        )

    def _process_answer(self, question: str, answer: str) -> None:
        """
        Process a user's answer.

        Args:
            question: The question that was asked
            answer: The user's answer
        """
        logger.debug(f"Processing answer: {answer[:100]}")

        # Add to conversation history
        self.context.add_message("user", answer)

        # Store in collector
        # This is simplified - will be enhanced in Phase 2
        topic = self._infer_topic_from_question(question)
        self.collector.add_question_to_history(topic, question, answer)

        # Mark category complete if enough info gathered for it
        if topic and not self._more_questions_needed(topic):
            self.collector.mark_category_complete(topic)

        logger.info(f"Added answer for topic: {topic}")

    def _infer_topic_from_question(self, question: str) -> str:
        """Infer the topic from a question."""
        question_lower = question.lower()

        if any(
            kw in question_lower
            for kw in ["flow", "reynolds", "turbulence", "compressible"]
        ):
            return InformationCategory.PHYSICS.value
        elif any(kw in question_lower for kw in ["geometry", "dimension", "size"]):
            return InformationCategory.GEOMETRY.value
        elif any(
            kw in question_lower
            for kw in ["boundary", "inlet", "outlet", "wall"]
        ):
            return InformationCategory.BOUNDARY_CONDITIONS.value
        elif any(kw in question_lower for kw in ["solver", "recommend"]):
            return InformationCategory.SOLVER.value
        else:
            return "general"

    def _more_questions_needed(self, topic: str) -> bool:
        """Check if more questions are needed for a topic."""
        # Simplified logic - will be enhanced with LLM decision
        questions_for_topic = self.collector.get_questions_by_topic(topic)
        return len(questions_for_topic) < 2  # Ask at least 2 per topic

    def _should_transition_to_summary(self) -> bool:
        """Check if we should transition to summary."""
        # Transition if we have enough questions
        return self.question_count >= 5  # Minimum questions before summary

    def _handle_command(self, user_input: str) -> None:
        """
        Handle a command from the user.

        Args:
            user_input: The command input
        """
        result = self.command_handler.parse_command(user_input)
        if not result:
            return

        command, argument = result
        response = self.command_handler.execute_command(command, argument)
        action = response.get("action")

        if action == "skip":
            console.print("[yellow]Skipped to next topic[/yellow]")
            self.question_count += 1

        elif action == "back":
            self._show_back_options(response)

        elif action == "summary":
            self.transition_to_summary()
            self.state = WorkflowState.REVIEWING_SUMMARY

        elif action == "edit":
            self._handle_edit_command(response.get("field"))

        elif action == "done":
            if response.get("is_complete"):
                console.print(
                    "[green]✓ All required information collected![/green]"
                )
                self.transition_to_summary()
            else:
                missing = response.get("missing", [])
                console.print(
                    "[yellow]Still need:[/yellow] " + ", ".join(missing[:3])
                )

        elif action == "generate":
            self.transition_to_summary()

        elif action == "exit":
            self.state = WorkflowState.COMPLETE
            console.print("[cyan]Goodbye![/cyan]")

        elif action == "help":
            self._show_help(response)

        elif action == "error":
            console.print(f"[red]{response.get('message')}[/red]")

    def _show_back_options(self, response: Dict[str, Any]) -> None:
        """Show options for editing previous answers."""
        if not response.get("found_history"):
            console.print("[yellow]No previous questions to edit[/yellow]")
            return

        history = response.get("history", [])
        console.print(
            Panel(
                f"Found {len(history)} previous questions. Type 'edit <topic>' to modify a topic.",
                title="Back",
                style="yellow",
            )
        )

    def _handle_edit_command(self, field: Optional[str]) -> None:
        """Handle field editing."""
        if not field:
            console.print("[yellow]Usage: edit <field> (e.g., edit physics.flow_type)[/yellow]")
            return

        console.print(f"[cyan]Editing: {field}[/cyan]")
        # More detailed edit handling will be implemented in Phase 3
        # For now, just acknowledge
        self.collector.mark_category_incomplete(field.split(".")[0])

    def _show_help(self, response: Dict[str, Any]) -> None:
        """Display help information."""
        commands = response.get("commands", {})
        help_text = "Available Commands:\n\n"
        for cmd, description in commands.items():
            help_text += f"  [bold]{cmd}[/bold] - {description}\n"

        console.print(Panel(help_text, title="Help", style="cyan"))

    def transition_to_summary(self) -> None:
        """Transition to the summary review state."""
        self.state = WorkflowState.REVIEWING_SUMMARY
        logger.info("Transitioning to summary review")
        self._display_summary()

    def _display_summary(self) -> None:
        """Display a summary of collected information."""
        status = self.collector.get_completion_status()
        data = self.collector.case_info["collected_data"]

        # Build a visual summary
        table = Table(title="Case Configuration Summary", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        for category, is_complete in status.items():
            status_icon = "✓" if is_complete else "!"
            status_text = f"[green]{status_icon} Complete[/green]" if is_complete else "[yellow]! Incomplete[/yellow]"

            # Get brief description of what was collected
            category_data = data.get(category, {})
            if category_data:
                details = ", ".join(
                    f"{k}={v}" for k, v in list(category_data.items())[:2]
                )[:40]
            else:
                details = "No data"

            table.add_row(category.replace("_", " ").title(), status_text, details)

        console.print(table)

        # Show missing required fields
        missing = self.collector.get_missing_required()
        if missing:
            console.print(
                Panel(
                    f"Missing required: {', '.join(missing)}",
                    style="yellow",
                    title="Required Fields",
                )
            )

        # Show next steps
        console.print(
            Panel(
                "Use [bold]edit <field>[/bold] to modify a value\n"
                "Use [bold]generate[/bold] to proceed to case generation\n"
                "Use [bold]back[/bold] to continue answering questions",
                title="Next Steps",
                style="cyan",
            )
        )

    def get_case_info(self) -> Dict[str, Any]:
        """Get the collected case information."""
        return self.collector.case_info.copy()

    def is_workflow_complete(self) -> bool:
        """Check if the workflow is complete."""
        return self.state == WorkflowState.COMPLETE
