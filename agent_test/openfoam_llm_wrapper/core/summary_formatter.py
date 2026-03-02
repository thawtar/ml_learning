"""Format and display case information summaries."""

from typing import Dict, Any, List
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def format_summary_for_display(case_info: Dict[str, Any]) -> str:
    """
    Create a human-readable summary for terminal display.

    Args:
        case_info: The collected case information dictionary

    Returns:
        Formatted string suitable for printing
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("OPENFOAM CASE CONFIGURATION SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Process each category
    collection_status = case_info.get("collection_status", {})
    collected_data = case_info.get("collected_data", {})

    for category, is_complete in collection_status.items():
        status_icon = "✓" if is_complete else "!"
        status_text = "COMPLETE" if is_complete else "INCOMPLETE"

        lines.append(f"\n[{status_icon}] {category.upper().replace('_', ' ')} - {status_text}")
        lines.append("-" * 60)

        # Show data for this category
        category_data = collected_data.get(category, {})

        if not category_data:
            lines.append("  (No data collected)")
        else:
            for key, value in category_data.items():
                # Format the value nicely
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    - {k}: {v}")
                elif isinstance(value, list):
                    lines.append(f"  {key}:")
                    for item in value:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {key}: {value}")

    # Question history
    question_history = case_info.get("question_history", [])
    if question_history:
        lines.append("\n" + "=" * 60)
        lines.append("QUESTION HISTORY")
        lines.append("=" * 60)
        for i, record in enumerate(question_history, 1):
            lines.append(f"\nQ{i}. {record.get('question', 'Unknown')}")
            lines.append(f"   A: {record.get('answer', 'No answer')}")

    return "\n".join(lines)


def format_summary_for_llm(case_info: Dict[str, Any]) -> str:
    """
    Create a summary formatted for LLM consumption.

    Args:
        case_info: The collected case information dictionary

    Returns:
        Formatted string for LLM processing
    """
    lines = []

    # Simple, LLM-friendly format
    lines.append("CASE CONFIGURATION:")
    lines.append("")

    collected_data = case_info.get("collected_data", {})

    # Physics
    physics = collected_data.get("physics", {})
    if physics:
        lines.append("Physics:")
        for key, value in physics.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

    # Geometry
    geometry = collected_data.get("geometry", {})
    if geometry:
        lines.append("Geometry:")
        for key, value in geometry.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

    # Boundary Conditions
    bcs = collected_data.get("boundary_conditions", {})
    if bcs:
        lines.append("Boundary Conditions:")
        for key, value in bcs.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  - {key}: {value}")
        lines.append("")

    # Solver
    solver = collected_data.get("solver", {})
    if solver:
        lines.append("Solver Configuration:")
        for key, value in solver.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

    # Other categories
    for category in [
        "fluid_properties",
        "mesh",
        "simulation_goals",
        "advanced",
    ]:
        data = collected_data.get(category, {})
        if data:
            lines.append(f"{category.replace('_', ' ').title()}:")
            for key, value in data.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

    return "\n".join(lines)


def create_case_description(case_info: Dict[str, Any]) -> str:
    """
    Convert structured case_info into a natural language description.

    This is used to create the comprehensive case description that's passed
    to the LLM for file generation.

    Args:
        case_info: The collected case information dictionary

    Returns:
        Natural language description of the case
    """
    lines = []
    collected_data = case_info.get("collected_data", {})

    # Physics section
    physics = collected_data.get("physics", {})
    if physics:
        flow_type = physics.get("flow_type", "unknown")
        time_type = physics.get("time_type", "unknown")
        reynolds = physics.get("reynolds_number", "not specified")
        turbulence = physics.get("turbulence_model", "not specified")

        lines.append(
            f"This is a {time_type} {flow_type} flow simulation "
            f"with Reynolds number approximately {reynolds}. "
        )

        if turbulence != "not specified" and turbulence.lower() != "laminar":
            lines.append(f"Turbulence modeling: {turbulence}. ")

        special = physics.get("special_physics", [])
        if special:
            if isinstance(special, list):
                lines.append(f"Special physics: {', '.join(special)}. ")
            else:
                lines.append(f"Special physics: {special}. ")

    # Geometry section
    geometry = collected_data.get("geometry", {})
    if geometry:
        description = geometry.get("description", "")
        if description:
            lines.append(f"Geometry: {description} ")

        dimension = geometry.get("dimension", "")
        if dimension:
            lines.append(f"({dimension} simulation) ")

    lines.append("")

    # Boundary conditions section
    bcs = collected_data.get("boundary_conditions", {})
    if bcs:
        patches = bcs.get("patches", {})
        if patches:
            lines.append("Boundary conditions:")

            for patch_name, patch_info in patches.items():
                if isinstance(patch_info, dict):
                    patch_type = patch_info.get("type", "not specified")
                    field_values = patch_info.get("field_values", {})

                    lines.append(f"  - {patch_name}: {patch_type}")
                    for field, value in field_values.items():
                        lines.append(f"    ({field}: {value})")
                else:
                    lines.append(f"  - {patch_name}: {patch_info}")

    lines.append("")

    # Fluid properties section
    fluid = collected_data.get("fluid_properties", {})
    if fluid:
        lines.append("Fluid properties:")
        density = fluid.get("density")
        viscosity = fluid.get("viscosity")

        if density:
            lines.append(f"  - Density: {density} kg/m³")
        if viscosity:
            lines.append(f"  - Viscosity: {viscosity} Pa·s")

    lines.append("")

    # Solver section
    solver = collected_data.get("solver", {})
    if solver:
        solver_name = solver.get("solver", "not specified")
        reason = solver.get("reason", "")

        lines.append(f"Solver: {solver_name}")
        if reason:
            lines.append(f"Reason: {reason}")

    lines.append("")

    # Mesh section
    mesh = collected_data.get("mesh", {})
    if mesh:
        lines.append("Mesh requirements:")
        mesh_type = mesh.get("type")
        if mesh_type:
            lines.append(f"  - Type: {mesh_type}")

        cell_count = mesh.get("cell_count_estimate")
        if cell_count:
            lines.append(f"  - Estimated cells: {cell_count}")

    lines.append("")

    # Simulation goals section
    goals = collected_data.get("simulation_goals", {})
    if goals:
        lines.append("Simulation goals:")
        targets = goals.get("target_outputs", [])
        if targets:
            lines.append(f"  - Target outputs: {', '.join(targets)}")

        convergence = goals.get("convergence_criteria")
        if convergence:
            lines.append(f"  - Convergence criteria: {convergence}")

    return "\n".join(lines)


def display_category_status(case_info: Dict[str, Any]) -> None:
    """
    Display the status of information collection using a table.

    Args:
        case_info: The collected case information dictionary
    """
    status = case_info.get("collection_status", {})
    data = case_info.get("collected_data", {})

    table = Table(title="Information Collection Status", show_header=True)
    table.add_column("Category", style="cyan", width=25)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Items Collected", width=30)

    for category, is_complete in status.items():
        # Create status badge
        if is_complete:
            status_badge = Text("✓ Complete", style="green")
        else:
            status_badge = Text("! Incomplete", style="yellow")

        # Count items in category
        category_data = data.get(category, {})
        item_count = len(category_data) if isinstance(category_data, dict) else 0

        # Format category name
        category_display = category.replace("_", " ").title()

        table.add_row(
            category_display, status_badge, f"{item_count} item(s)"
        )

    console.print(table)


def display_question_history(case_info: Dict[str, Any]) -> None:
    """
    Display the question-answer history.

    Args:
        case_info: The collected case information dictionary
    """
    history = case_info.get("question_history", [])

    if not history:
        console.print("[yellow]No questions asked yet[/yellow]")
        return

    lines = []
    lines.append("")
    for i, record in enumerate(history, 1):
        topic = record.get("topic", "General").upper()
        question = record.get("question", "Unknown question")
        answer = record.get("answer", "No answer")

        # Truncate long answers
        if len(answer) > 60:
            answer = answer[:60] + "..."

        lines.append(f"Q{i}. [{topic}] {question}")
        lines.append(f"    A: {answer}")
        lines.append("")

    panel_text = "\n".join(lines)
    console.print(Panel(panel_text, title="Question History", style="cyan"))


def get_missing_required_fields(case_info: Dict[str, Any]) -> List[str]:
    """
    Get list of missing required fields.

    Args:
        case_info: The collected case information dictionary

    Returns:
        List of missing required fields
    """
    # Define minimum required information
    REQUIRED_FIELDS = {
        "physics": ["flow_type", "time_type"],
        "geometry": ["description"],
        "boundary_conditions": ["patches"],
        "solver": ["solver"],
    }

    missing = []
    collected_data = case_info.get("collected_data", {})

    for category, required in REQUIRED_FIELDS.items():
        category_data = collected_data.get(category, {})

        for field in required:
            if field not in category_data or not category_data[field]:
                missing.append(f"{category.replace('_', ' ')}: {field}")

    return missing


def display_edit_menu(case_info: Dict[str, Any]) -> None:
    """
    Display a menu for editing case information.

    Args:
        case_info: The collected case information dictionary
    """
    collected_data = case_info.get("collected_data", {})

    lines = [
        "You can edit the following fields:",
        "",
    ]

    for category, data in collected_data.items():
        if data:
            lines.append(f"[bold]{category.replace('_', ' ').title()}:[/bold]")
            for key in data.keys():
                lines.append(f"  • edit {category}.{key}")
            lines.append("")

    console.print("\n".join(lines))
