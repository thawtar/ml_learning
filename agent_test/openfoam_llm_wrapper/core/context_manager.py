"""Manage conversation context and session state."""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Manages the conversation history and context."""

    messages: List[Message] = field(default_factory=list)
    case_info: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def add_message(
        self, role: str, content: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata about the message
        """
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        logger.debug(f"Added {role} message to context")

    def get_conversation_history(self, limit: int | None = None) -> List[Dict[str, str]]:
        """
        Get conversation history in LLM-friendly format.

        Args:
            limit: Optional limit on number of recent messages to return

        Returns:
            List of dictionaries with role and content
        """
        messages = self.messages if limit is None else self.messages[-limit:]
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_history(self) -> None:
        """Clear conversation history while preserving case info."""
        self.messages.clear()
        logger.info("Cleared conversation history")

    def set_case_info(self, key: str, value: Any) -> None:
        """Store case-related information."""
        self.case_info[key] = value
        logger.debug(f"Set case info: {key}")

    def get_case_info(self, key: str, default: Any = None) -> Any:
        """Retrieve case-related information."""
        return self.case_info.get(key, default)

    def update_nested_field(self, path: str, value: Any) -> None:
        """
        Update a nested field using dot notation.

        Args:
            path: Dot-notation path (e.g., "collected_data.physics.flow_type")
            value: Value to set
        """
        parts = path.split(".")
        if len(parts) < 2:
            logger.warning(f"Invalid path format: {path}")
            return

        # Navigate to the nested location
        current = self.case_info
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value
        logger.debug(f"Updated nested field: {path} = {value}")

    def get_nested_field(self, path: str, default: Any = None) -> Any:
        """
        Get a nested field using dot notation.

        Args:
            path: Dot-notation path (e.g., "collected_data.physics.flow_type")
            default: Default value if not found

        Returns:
            Field value or default
        """
        parts = path.split(".")
        current = self.case_info

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default

        return current if current is not None else default

    def initialize_case_info(self) -> None:
        """Initialize case_info with proper structure."""
        if not self.case_info:
            self.case_info = {
                "workflow_state": "initialization",
                "collection_status": {
                    "physics": False,
                    "geometry": False,
                    "boundary_conditions": False,
                    "fluid_properties": False,
                    "solver": False,
                    "mesh": False,
                    "simulation_goals": False,
                    "advanced": False,
                },
                "collected_data": {
                    "physics": {},
                    "geometry": {},
                    "boundary_conditions": {},
                    "fluid_properties": {},
                    "solver": {},
                    "mesh": {},
                    "simulation_goals": {},
                    "advanced": {},
                },
                "question_history": [],
                "user_edits": [],
            }
            logger.info("Initialized case_info structure")

    def mark_category_complete(self, category: str) -> None:
        """
        Mark an information category as complete.

        Args:
            category: Category name (e.g., "physics")
        """
        if "collection_status" not in self.case_info:
            self.case_info["collection_status"] = {}

        self.case_info["collection_status"][category] = True
        logger.debug(f"Marked {category} as complete")

    def mark_category_incomplete(self, category: str) -> None:
        """
        Mark an information category as incomplete.

        Args:
            category: Category name
        """
        if "collection_status" not in self.case_info:
            self.case_info["collection_status"] = {}

        self.case_info["collection_status"][category] = False
        logger.debug(f"Marked {category} as incomplete")

    def get_completion_status(self) -> Dict[str, bool]:
        """
        Get completion status of all categories.

        Returns:
            Dictionary mapping category to completion status
        """
        if "collection_status" not in self.case_info:
            return {}

        return self.case_info["collection_status"].copy()

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
        if "question_history" not in self.case_info:
            self.case_info["question_history"] = []

        record = {
            "topic": topic,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }
        self.case_info["question_history"].append(record)
        logger.debug(f"Added question to history: {topic}")

    def get_question_history(self) -> List[Dict[str, Any]]:
        """
        Get the question history.

        Returns:
            List of question-answer records
        """
        return self.case_info.get("question_history", []).copy()

    def get_questions_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get all questions for a specific topic.

        Args:
            topic: Topic name

        Returns:
            List of question records for that topic
        """
        history = self.case_info.get("question_history", [])
        return [q for q in history if q.get("topic") == topic]
