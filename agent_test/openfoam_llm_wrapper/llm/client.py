"""LLM API client abstraction for Claude and other providers."""

import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with LLM APIs.

    Currently supports Anthropic Claude API.
    Can be extended to support other providers (OpenAI, etc).
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider (uses env var if not provided)
            provider: LLM provider ("anthropic" or "openai")
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

        if not self.api_key:
            logger.warning(
                f"No {provider.upper()}_API_KEY found. LLM features will not work."
            )

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the underlying LLM client based on provider."""
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key) if self.api_key else None
            except ImportError:
                logger.error("anthropic package not installed")
                self.client = None
        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            except ImportError:
                logger.error("openai package not installed")
                self.client = None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_case_files(self, description: str, intent: str) -> dict:
        """
        Generate OpenFOAM case files based on description.

        Args:
            description: Natural language description of the case
            intent: The classified intent of the user

        Returns:
            Dictionary containing generated file contents
        """
        if not self.client:
            logger.warning("LLM client not initialized, returning placeholder response")
            return self._placeholder_response(intent)

        from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_case_generation_prompt(description, intent)

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content

            # Parse response into files
            from openfoam_llm_wrapper.llm.response_parser import ResponseParser

            parser = ResponseParser()
            files = parser.parse_case_files(response_text)
            logger.info(f"Generated {len(files)} files from LLM")
            return files

        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return self._placeholder_response(intent)

    def explain_error(self, error_message: str) -> str:
        """
        Explain an OpenFOAM error message.

        Args:
            error_message: The error message to explain

        Returns:
            Explanation of the error
        """
        if not self.client:
            return "LLM client not initialized. Cannot explain error."

        from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_error_explanation_prompt(error_message)

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return f"Could not explain error: {str(e)}"

    def recommend_solver(self, physics_description: str) -> str:
        """
        Recommend an appropriate OpenFOAM solver.

        Args:
            physics_description: Description of the simulation physics

        Returns:
            Solver recommendation with reasoning
        """
        if not self.client:
            return "LLM client not initialized. Cannot recommend solver."

        from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_solver_recommendation_prompt(physics_description)

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return f"Could not get recommendation: {str(e)}"

    def answer_question(self, question: str) -> str:
        """
        Answer a general question about OpenFOAM.

        Args:
            question: The question to answer

        Returns:
            Answer text
        """
        if not self.client:
            return "LLM client not initialized. Cannot answer question."

        from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_question_prompt(question)

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return f"Could not answer question: {str(e)}"

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """
        Conduct a multi-turn conversation with full conversation history.

        This method passes the complete conversation history to the LLM,
        enabling context-aware responses and follow-up questions.

        Args:
            messages: List of {"role": "user/assistant", "content": str}
            system_prompt: System instructions for LLM behavior
            max_tokens: Maximum response length

        Returns:
            LLM response string
        """
        if not self.client:
            logger.warning("LLM client not initialized")
            return "LLM client not initialized. Cannot process request."

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                response_text = response.content[0].text
            else:  # openai
                # For OpenAI, include system prompt in messages
                messages_with_system = [
                    {"role": "system", "content": system_prompt}
                ] + messages

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=max_tokens,
                    messages=messages_with_system,
                )
                response_text = response.choices[0].message.content

            logger.debug(f"LLM response: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"Error in chat_with_history: {str(e)}")
            return f"Error: {str(e)}"

    def interactive_case_questioning(
        self,
        conversation_history: List[Dict[str, str]],
        collection_status: Dict[str, bool],
        collected_data: Dict[str, Any],
        question_history: List[Dict[str, Any]],
        knowledge_context: str = "",
    ) -> str:
        """
        Get the next question for interactive case generation.

        Uses the current collection status and previous answers to determine
        what to ask next, enabling intelligent question sequencing.

        Args:
            conversation_history: Full conversation history with role/content
            collection_status: Dict showing which categories are complete
            collected_data: Dict of collected information by category
            question_history: List of previous question-answer pairs
            knowledge_context: Optional knowledge base context about solvers, BCs

        Returns:
            The next question to ask or completion signal
        """
        from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()
        system_prompt = prompt_builder.build_interactive_system_prompt(
            collection_status=collection_status,
            collected_data=collected_data,
            question_history=question_history,
            knowledge_base=knowledge_context,
        )

        logger.debug(
            f"Asking next question. Collection status: {collection_status}"
        )

        return self.chat_with_history(
            messages=conversation_history,
            system_prompt=system_prompt,
            max_tokens=1024,
        )

    @staticmethod
    def _placeholder_response(intent: str) -> dict:
        """Return a placeholder response when LLM is not available."""
        return {
            "system/controlDict": "# Placeholder controlDict\n# Configure your solver settings here",
            "system/fvSchemes": "# Placeholder fvSchemes\n# Define discretization schemes",
            "system/fvSolution": "# Placeholder fvSolution\n# Configure solver parameters",
            "0/U": "# Placeholder velocity field\n# Set initial conditions",
            "0/p": "# Placeholder pressure field\n# Set initial conditions",
        }
