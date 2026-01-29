"""Classify user intent from input text."""

from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Enumeration of supported user intents."""

    GENERATE_CASE = "generate_case"
    EXPLAIN_ERROR = "explain_error"
    RECOMMEND_SOLVER = "recommend_solver"
    GENERAL_QUESTION = "general_question"
    ANALYZE_MESH = "analyze_mesh"
    HELP = "help"


class IntentClassifier:
    """Classify user intent from natural language input."""

    # Keywords for intent detection
    INTENT_KEYWORDS = {
        Intent.GENERATE_CASE: [
            "generate",
            "create",
            "setup",
            "case",
            "simulation",
            "configure",
        ],
        Intent.EXPLAIN_ERROR: [
            "error",
            "error message",
            "failed",
            "explain",
            "what does",
            "mean",
        ],
        Intent.RECOMMEND_SOLVER: [
            "solver",
            "which solver",
            "recommend",
            "incompressible",
            "compressible",
        ],
        Intent.ANALYZE_MESH: [
            "mesh",
            "checkmesh",
            "quality",
            "refinement",
        ],
        Intent.HELP: ["help", "how do i", "how to", "tutorial"],
    }

    def classify(self, text: str) -> Intent:
        """
        Classify the intent of the input text.

        Args:
            text: User input text

        Returns:
            Detected Intent enum value
        """
        text_lower = text.lower()

        # Count keyword matches for each intent
        scores = {intent: 0 for intent in Intent}

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[intent] += 1

        # Return intent with highest score, default to GENERAL_QUESTION
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = Intent.GENERAL_QUESTION

        logger.debug(f"Intent classification scores: {scores}")
        logger.info(f"Classified intent: {best_intent.value}")

        return best_intent
