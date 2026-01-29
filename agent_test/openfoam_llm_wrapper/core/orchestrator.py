"""Main workflow orchestrator for OpenFOAM case generation and management."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the workflow between LLM, generators, and validators."""

    def __init__(self):
        """Initialize the orchestrator with required components."""
        # Import here to avoid circular dependencies
        from openfoam_llm_wrapper.llm.client import LLMClient
        from openfoam_llm_wrapper.core.intent_classifier import IntentClassifier
        from openfoam_llm_wrapper.generators.case_generator import CaseGenerator
        from openfoam_llm_wrapper.validators.syntax_validator import SyntaxValidator

        self.llm_client = LLMClient()
        self.intent_classifier = IntentClassifier()
        self.case_generator = CaseGenerator()
        self.syntax_validator = SyntaxValidator()

    def generate_case(
        self, description: str, output_dir: str = ".", case_name: str = "openfoam_case"
    ) -> Dict[str, Any]:
        """
        Generate a complete OpenFOAM case from a natural language description.

        Args:
            description: Natural language description of the simulation
            output_dir: Directory to output the case
            case_name: Name of the case directory to create

        Returns:
            Dictionary with case generation results including case_path
        """
        logger.info(f"Starting case generation for: {case_name}")

        # Step 1: Classify intent
        intent = self.intent_classifier.classify(description)
        logger.debug(f"Detected intent: {intent}")

        # Step 2: Get LLM response
        llm_response = self.llm_client.generate_case_files(description, intent)
        logger.debug(f"LLM response received: {len(llm_response)} files")

        # Step 3: Generate files
        case_path = self.case_generator.create_case(
            llm_response, output_dir, case_name
        )
        logger.info(f"Case created at: {case_path}")

        # Step 4: Validate
        validation_result = self.syntax_validator.validate_case(case_path)
        logger.info(f"Validation result: {validation_result}")

        return {
            "case_path": case_path,
            "validation": validation_result,
            "intent": intent,
        }

    def explain_error(self, error_message: str) -> str:
        """
        Explain an OpenFOAM error message using LLM.

        Args:
            error_message: The error message to explain

        Returns:
            Explanation of the error
        """
        logger.info("Explaining error message")
        explanation = self.llm_client.explain_error(error_message)
        return explanation

    def recommend_solver(self, physics_description: str) -> str:
        """
        Recommend appropriate OpenFOAM solver based on physics description.

        Args:
            physics_description: Description of the simulation physics

        Returns:
            Solver recommendation with reasoning
        """
        logger.info("Getting solver recommendation")
        recommendation = self.llm_client.recommend_solver(physics_description)
        return recommendation

    def generate_case_from_interactive(
        self,
        case_info: Dict[str, Any],
        output_dir: str = ".",
        case_name: str = "openfoam_case",
    ) -> Dict[str, Any]:
        """
        Generate an OpenFOAM case from structured interactive workflow data.

        This method takes the structured case information collected during
        the interactive workflow and generates the complete OpenFOAM case.

        Args:
            case_info: The collected case information dictionary
            output_dir: Directory to output the case
            case_name: Name of the case directory to create

        Returns:
            Dictionary with case generation results
        """
        logger.info(
            f"Starting case generation from interactive data for: {case_name}"
        )

        try:
            # Step 1: Build comprehensive case description from structured data
            from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

            prompt_builder = PromptBuilder()
            case_prompt = prompt_builder.build_final_case_prompt(case_info)

            logger.debug("Built comprehensive case description from structured data")

            # Step 2: Use LLM to generate case files
            response = self.llm_client.chat_with_history(
                messages=[{"role": "user", "content": case_prompt}],
                system_prompt="You are an expert OpenFOAM consultant. Generate complete, correct OpenFOAM case files.",
                max_tokens=4096,
            )

            # Step 3: Parse response into files
            from openfoam_llm_wrapper.llm.response_parser import ResponseParser

            parser = ResponseParser()
            files = parser.parse_case_files(response)
            logger.info(f"Generated {len(files)} files from LLM response")

            if not files:
                logger.warning("No files generated from LLM response")
                return {
                    "case_path": None,
                    "validation": {"is_valid": False, "errors": [
                        "No files generated from LLM response"
                    ]},
                    "case_info": case_info,
                }

            # Step 4: Create case directory and write files
            case_path = self.case_generator.create_case(
                files, output_dir, case_name
            )
            logger.info(f"Case created at: {case_path}")

            # Step 5: Validate the generated case
            validation_result = self.syntax_validator.validate_case(case_path)
            logger.info(f"Validation result: {validation_result}")

            return {
                "case_path": case_path,
                "validation": validation_result,
                "case_info": case_info,
                "files_generated": len(files),
            }

        except Exception as e:
            logger.error(f"Error in case generation: {str(e)}")
            return {
                "case_path": None,
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                },
                "case_info": case_info,
            }
