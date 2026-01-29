"""Build domain-specific prompts for the LLM."""

from typing import Dict, List, Any

import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Constructs prompts tailored for OpenFOAM-related tasks."""

    SYSTEM_PROMPT = """You are an expert OpenFOAM consultant with 20 years of CFD experience.
You help engineers set up simulations by generating correct OpenFOAM files and answering questions.

CRITICAL RULES:
1. All dictionary files must use correct OpenFOAM syntax (exact spacing, semicolons, brackets)
2. Patch names must be consistent across all files
3. Recommend conservative numerical settings for initial runs
4. Always explain non-obvious configuration choices
5. Validate physical reasonableness of parameters
6. When generating files, wrap each in <file path="..."> tags"""

    def build_case_generation_prompt(self, description: str, intent: str) -> str:
        """
        Build prompt for case file generation.

        Args:
            description: Natural language case description
            intent: The classified intent

        Returns:
            Formatted prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

Generate a complete OpenFOAM case structure for the following simulation:

{description}

Please generate the following files in <file> tags:
- system/controlDict (solver control parameters)
- system/fvSchemes (discretization schemes)
- system/fvSolution (linear solver settings)
- constant/transportProperties (fluid properties)
- 0/U (velocity field)
- 0/p (pressure field)

For each file, ensure:
1. Correct OpenFOAM syntax with proper formatting
2. Appropriate boundary condition types for the described geometry
3. Conservative solver tolerances and relaxation factors
4. Comments explaining key parameters

Format each file as:
<file path="path/to/file">
file contents here
</file>
"""

    def build_error_explanation_prompt(self, error_message: str) -> str:
        """
        Build prompt for error message explanation.

        Args:
            error_message: The OpenFOAM error message

        Returns:
            Formatted prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

I received the following OpenFOAM error message. Please explain:
1. What went wrong
2. Why it happened
3. How to fix it

Error message:
{error_message}

Provide a clear, concise explanation suitable for a CFD engineer."""

    def build_solver_recommendation_prompt(self, physics_description: str) -> str:
        """
        Build prompt for solver recommendation.

        Args:
            physics_description: Description of the simulation physics

        Returns:
            Formatted prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

Based on the following simulation physics, recommend an appropriate OpenFOAM solver:

{physics_description}

Please provide:
1. Recommended solver(s)
2. Why this solver is appropriate
3. Key settings to configure
4. Any limitations or considerations

Consider these aspects:
- Flow regime (incompressible/compressible, subsonic/supersonic)
- Time dependency (steady/transient)
- Turbulence modeling needed
- Special physics (multiphase, combustion, heat transfer, etc)"""

    def build_question_prompt(self, question: str) -> str:
        """
        Build prompt for general questions.

        Args:
            question: The user's question

        Returns:
            Formatted prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

Answer the following OpenFOAM-related question:

{question}

Provide a practical, actionable answer. If applicable, include relevant OpenFOAM file examples or code."""

    def build_interactive_system_prompt(
        self,
        collection_status: Dict[str, bool],
        collected_data: Dict[str, Any],
        question_history: List[Dict[str, Any]],
        knowledge_base: str = "",
    ) -> str:
        """
        Build system prompt for interactive case generation.

        This prompt guides the LLM to ask intelligent follow-up questions
        based on what has already been collected.

        Args:
            collection_status: Dict showing completion status of categories
            collected_data: Dict of collected information
            question_history: List of previous Q&A pairs
            knowledge_base: Optional knowledge base context

        Returns:
            System prompt string
        """
        # Format collection status
        status_text = self._format_collection_status(collection_status)

        # Format collected data
        data_text = self._format_collected_data(collected_data)

        # Format question history
        history_text = self._format_question_history(question_history)

        prompt = f"""You are an expert OpenFOAM consultant conducting an interactive case setup interview with an experienced CFD engineer.

YOUR ROLE:
- Ask ONE expert-level technical question at a time
- Be concise and precise (assume CFD knowledge)
- Build intelligently upon previous answers
- Determine what information is still needed for a complete case setup
- When you have sufficient information for case generation, respond with exactly: READY_TO_GENERATE

INFORMATION CATEGORIES TO GATHER:
1. Physics: flow type (incompressible/compressible), time dependency (steady/transient), Reynolds number, turbulence model, special physics
2. Geometry: description, dimensions, characteristic length, 2D/3D, symmetries
3. Boundary Conditions: inlet, outlet, wall specifications with values
4. Solver: OpenFOAM solver selection (simpleFoam, pimpleFoam, etc)
5. Fluid Properties: density, viscosity, thermal properties (if needed)
6. Mesh: generation method, refinement requirements, cell count estimate
7. Simulation Goals: what to compute (drag, pressure drop, etc), convergence criteria
8. Advanced: optional custom schemes, relaxation factors, solver settings

CURRENT COLLECTION STATUS:
{status_text}

WHAT'S BEEN COLLECTED SO FAR:
{data_text}

PREVIOUS QUESTIONS ASKED:
{history_text}

{knowledge_base}

SPECIAL HANDLING:
- If user says "skip": Acknowledge and move to next topic without that information
- If user says "back": Help them revise a previous answer
- If user says "summary": Provide a summary of collected information
- If user says "done" or "ready": Check if we have enough for generation, respond with READY_TO_GENERATE if yes, or explain what's missing if no

YOUR NEXT QUESTION:
Based on what's been collected above and what's still needed, ask the next most important technical question for this simulation setup. Ask only ONE question. Be specific and expert-oriented."""

        return prompt

    def _format_collection_status(self, status: Dict[str, bool]) -> str:
        """Format collection status for display in prompt."""
        lines = []
        for category, is_complete in status.items():
            icon = "✓" if is_complete else "!"
            lines.append(f"  [{icon}] {category.replace('_', ' ').title()}")

        return "\n".join(lines) if lines else "  (No data collected yet)"

    def _format_collected_data(self, data: Dict[str, Any]) -> str:
        """Format collected data for display in prompt."""
        lines = []

        for category, category_data in data.items():
            if not category_data:
                continue

            lines.append(f"\n{category.upper().replace('_', ' ')}:")

            if isinstance(category_data, dict):
                for key, value in category_data.items():
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:60] + "..."

                    lines.append(f"  - {key}: {value_str}")
            else:
                lines.append(f"  - {category_data}")

        return "\n".join(lines) if lines else "  (No data collected yet)"

    def _format_question_history(self, history: List[Dict[str, Any]]) -> str:
        """Format question history for display in prompt."""
        if not history:
            return "  (No previous questions)"

        lines = []
        for i, record in enumerate(history[-5:], 1):  # Show last 5 questions
            topic = record.get("topic", "General")
            question = record.get("question", "Unknown")
            answer = record.get("answer", "No answer")

            # Truncate long answers
            if len(answer) > 50:
                answer = answer[:50] + "..."

            lines.append(f"  Q{i}. [{topic.upper()}] {question}")
            lines.append(f"       A: {answer}")

        return "\n".join(lines)

    def build_final_case_prompt(
        self, case_info: Dict[str, Any]
    ) -> str:
        """
        Build the final prompt for case generation from structured data.

        Converts the structured case_info into a comprehensive natural language
        description for the case generation engine.

        Args:
            case_info: The collected case information dictionary

        Returns:
            Comprehensive case description prompt
        """
        from openfoam_llm_wrapper.core.summary_formatter import (
            create_case_description,
        )

        # Convert structured data to natural language
        description = create_case_description(case_info)

        return f"""{self.SYSTEM_PROMPT}

Generate a complete OpenFOAM case structure for the following simulation:

{description}

Please generate the following files in <file> tags:
- system/controlDict (solver control parameters)
- system/fvSchemes (discretization schemes)
- system/fvSolution (linear solver settings and algorithm parameters)
- constant/transportProperties (fluid properties)
- 0/U (velocity field with boundary conditions)
- 0/p (pressure field with boundary conditions)

For each file, ensure:
1. Correct OpenFOAM syntax with proper formatting
2. Appropriate boundary condition types for the described geometry
3. Conservative solver tolerances and relaxation factors
4. Comments explaining key parameters and assumptions

Format each file as:
<file path="path/to/file">
file contents here
</file>
"""

    def build_knowledge_base_context(self) -> str:
        """
        Build context about available solvers and boundary conditions.

        This is injected into the interactive prompt to help the LLM
        make informed solver and BC recommendations.

        Returns:
            Knowledge base context string
        """
        from openfoam_llm_wrapper.knowledge.solvers import (
            SOLVERS,
            list_all_solvers,
        )
        from openfoam_llm_wrapper.knowledge.boundary_conditions import (
            BOUNDARY_CONDITIONS,
            list_all_bcs,
        )

        lines = ["\nAVAILABLE SOLVERS:"]

        for solver_name in list_all_solvers():
            solver = SOLVERS.get(solver_name)
            if solver:
                lines.append(
                    f"  • {solver_name} ({solver.category}, {solver.time_type}): "
                    f"{solver.description}"
                )

        lines.append("\nAVAILABLE BOUNDARY CONDITIONS:")

        for bc_name in list_all_bcs():
            bc = BOUNDARY_CONDITIONS.get(bc_name)
            if bc:
                # Show first use case
                use_case = bc.typical_use_cases[0] if bc.typical_use_cases else "General"
                lines.append(f"  • {bc_name}: {bc.description} ({use_case})")

        return "\n".join(lines)
