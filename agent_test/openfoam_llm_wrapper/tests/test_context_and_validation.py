"""Unit tests for context manager and validation components."""

import pytest
from datetime import datetime

from openfoam_llm_wrapper.core.context_manager import (
    ConversationContext,
    Message,
)
from openfoam_llm_wrapper.knowledge.information_schema import (
    FieldValidator,
    FieldType,
    validate_field,
    validate_category,
    validate_case_info,
    get_field_description,
    get_valid_values,
)


class TestMessage:
    """Test suite for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_metadata(self):
        """Test creating message with metadata."""
        metadata = {"source": "test"}
        msg = Message(role="assistant", content="Response", metadata=metadata)

        assert msg.metadata == metadata

    def test_message_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")

        assert msg1.timestamp is not None
        assert msg2.timestamp is not None
        # They might be very close but should be different objects
        assert msg1.timestamp <= msg2.timestamp


class TestConversationContext:
    """Test suite for ConversationContext class."""

    @pytest.fixture
    def context(self):
        """Create a fresh context for each test."""
        return ConversationContext()

    def test_initialization(self, context):
        """Test context initializes with empty state."""
        assert len(context.messages) == 0
        assert isinstance(context.case_info, dict)

    def test_add_message(self, context):
        """Test adding a message."""
        context.add_message("user", "Test message")

        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Test message"

    def test_add_multiple_messages(self, context):
        """Test adding multiple messages."""
        context.add_message("user", "Q1")
        context.add_message("assistant", "A1")
        context.add_message("user", "Q2")

        assert len(context.messages) == 3
        assert context.messages[0].role == "user"
        assert context.messages[1].role == "assistant"

    def test_get_conversation_history(self, context):
        """Test getting conversation history in LLM format."""
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there")

        history = context.get_conversation_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there"

    def test_get_conversation_history_with_limit(self, context):
        """Test getting conversation history with limit."""
        for i in range(5):
            context.add_message("user", f"Message {i}")

        history = context.get_conversation_history(limit=3)

        assert len(history) == 3
        assert "Message 2" in history[-1]["content"]

    def test_clear_history(self, context):
        """Test clearing conversation history."""
        context.add_message("user", "Test")
        context.set_case_info("test_key", "test_value")

        context.clear_history()

        assert len(context.messages) == 0
        # case_info should be preserved
        assert context.get_case_info("test_key") == "test_value"

    def test_set_case_info(self, context):
        """Test setting case info."""
        context.set_case_info("flow_type", "incompressible")

        assert context.get_case_info("flow_type") == "incompressible"

    def test_get_case_info_with_default(self, context):
        """Test getting case info with default."""
        result = context.get_case_info("nonexistent", "default")

        assert result == "default"

    def test_update_nested_field(self, context):
        """Test updating nested field."""
        context.initialize_case_info()
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")

        value = context.get_nested_field("collected_data.physics.flow_type")
        assert value == "incompressible"

    def test_get_nested_field_with_default(self, context):
        """Test getting nested field with default."""
        value = context.get_nested_field("nonexistent.path", "default_value")

        assert value == "default_value"

    def test_initialize_case_info(self, context):
        """Test initializing case_info structure."""
        context.initialize_case_info()

        assert "collection_status" in context.case_info
        assert "collected_data" in context.case_info
        assert "question_history" in context.case_info

    def test_mark_category_complete(self, context):
        """Test marking category complete."""
        context.initialize_case_info()
        context.mark_category_complete("physics")

        assert context.case_info["collection_status"]["physics"] is True

    def test_mark_category_incomplete(self, context):
        """Test marking category incomplete."""
        context.initialize_case_info()
        context.mark_category_complete("physics")
        context.mark_category_incomplete("physics")

        assert context.case_info["collection_status"]["physics"] is False

    def test_get_completion_status(self, context):
        """Test getting completion status."""
        context.initialize_case_info()
        context.mark_category_complete("physics")
        context.mark_category_complete("geometry")

        status = context.get_completion_status()

        assert status["physics"] is True
        assert status["geometry"] is True
        assert status["solver"] is False

    def test_add_question_to_history(self, context):
        """Test adding question to history."""
        context.initialize_case_info()
        context.add_question_to_history("physics", "What type?", "incompressible")

        history = context.case_info["question_history"]

        assert len(history) == 1
        assert history[0]["topic"] == "physics"
        assert history[0]["question"] == "What type?"
        assert history[0]["answer"] == "incompressible"

    def test_get_question_history(self, context):
        """Test getting question history."""
        context.initialize_case_info()
        context.add_question_to_history("physics", "Q1", "A1")
        context.add_question_to_history("geometry", "Q2", "A2")

        history = context.get_question_history()

        assert len(history) == 2

    def test_get_questions_by_topic(self, context):
        """Test getting questions by topic."""
        context.initialize_case_info()
        context.add_question_to_history("physics", "Q1", "A1")
        context.add_question_to_history("physics", "Q2", "A2")
        context.add_question_to_history("geometry", "Q3", "A3")

        physics_q = context.get_questions_by_topic("physics")

        assert len(physics_q) == 2
        assert all(q["topic"] == "physics" for q in physics_q)


class TestFieldValidator:
    """Test suite for FieldValidator class."""

    def test_string_validation_success(self):
        """Test successful string validation."""
        validator = FieldValidator(FieldType.STRING, required=True)
        is_valid, error = validator.validate("test string")

        assert is_valid is True
        assert error is None

    def test_string_validation_failure(self):
        """Test failed string validation (wrong type)."""
        validator = FieldValidator(FieldType.STRING, required=True)
        is_valid, error = validator.validate(123)

        assert is_valid is False
        assert error is not None

    def test_float_validation_success(self):
        """Test successful float validation."""
        validator = FieldValidator(FieldType.FLOAT, required=True)
        is_valid, error = validator.validate(3.14)

        assert is_valid is True

    def test_float_validation_from_string(self):
        """Test float validation from string."""
        validator = FieldValidator(FieldType.FLOAT, required=True)
        is_valid, error = validator.validate("3.14")

        assert is_valid is True

    def test_integer_validation_success(self):
        """Test successful integer validation."""
        validator = FieldValidator(FieldType.INTEGER, required=True)
        is_valid, error = validator.validate(42)

        assert is_valid is True

    def test_boolean_validation_success(self):
        """Test successful boolean validation."""
        validator = FieldValidator(FieldType.BOOLEAN, required=True)
        is_valid, error = validator.validate(True)

        assert is_valid is True

    def test_list_validation_success(self):
        """Test successful list validation."""
        validator = FieldValidator(FieldType.LIST, required=True)
        is_valid, error = validator.validate([1, 2, 3])

        assert is_valid is True

    def test_dict_validation_success(self):
        """Test successful dict validation."""
        validator = FieldValidator(FieldType.DICT, required=True)
        is_valid, error = validator.validate({"key": "value"})

        assert is_valid is True

    def test_required_field_missing(self):
        """Test required field missing validation."""
        validator = FieldValidator(FieldType.STRING, required=True)
        is_valid, error = validator.validate(None)

        assert is_valid is False
        assert "Required" in error

    def test_optional_field_missing(self):
        """Test optional field missing validation."""
        validator = FieldValidator(FieldType.STRING, required=False)
        is_valid, error = validator.validate(None)

        assert is_valid is True

    def test_valid_values_check(self):
        """Test valid values checking."""
        validator = FieldValidator(
            FieldType.STRING,
            valid_values=["incompressible", "compressible"]
        )
        is_valid, error = validator.validate("incompressible")

        assert is_valid is True

    def test_invalid_value_check(self):
        """Test invalid value detection."""
        validator = FieldValidator(
            FieldType.STRING,
            valid_values=["incompressible", "compressible"]
        )
        is_valid, error = validator.validate("invalid")

        assert is_valid is False
        assert "not in valid options" in error


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_field_success(self):
        """Test successful field validation."""
        is_valid, error = validate_field("physics", "flow_type", "incompressible")

        assert is_valid is True

    def test_validate_field_invalid_value(self):
        """Test field validation with invalid value."""
        is_valid, error = validate_field("physics", "flow_type", "invalid_flow")

        assert is_valid is False

    def test_validate_field_unknown_category(self):
        """Test validation with unknown category."""
        is_valid, error = validate_field("unknown", "field", "value")

        assert is_valid is False

    def test_validate_category_complete(self):
        """Test validating complete category."""
        data = {
            "flow_type": "incompressible",
            "time_type": "steady",
            "reynolds_number": 100000,
            "turbulence_model": "kOmegaSST",
            "special_physics": ["heat_transfer"]
        }

        result = validate_category("physics", data)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_category_incomplete(self):
        """Test validating incomplete category."""
        data = {
            "flow_type": "incompressible"
            # Missing time_type which is required
        }

        result = validate_category("physics", data)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_case_info_complete(self):
        """Test validating complete case info."""
        case_info = {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady"
                },
                "geometry": {"description": "pipe"},
                "boundary_conditions": {"patches": {}},
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {}
            }
        }

        result = validate_case_info(case_info)

        assert result.is_valid is True

    def test_validate_case_info_missing_category(self):
        """Test validating case info with missing category."""
        case_info = {
            "collected_data": {
                # Missing several required categories
                "physics": {"flow_type": "incompressible", "time_type": "steady"}
            }
        }

        result = validate_case_info(case_info)

        assert result.is_valid is False

    def test_get_field_description(self):
        """Test getting field description."""
        description = get_field_description("physics", "flow_type")

        assert description is not None
        assert len(description) > 0

    def test_get_field_description_unknown(self):
        """Test getting description for unknown field."""
        description = get_field_description("unknown", "field")

        assert description is None

    def test_get_valid_values(self):
        """Test getting valid values for field."""
        valid_values = get_valid_values("physics", "flow_type")

        assert valid_values is not None
        assert "incompressible" in valid_values
        assert "compressible" in valid_values

    def test_get_valid_values_no_constraint(self):
        """Test getting valid values for unconstrained field."""
        valid_values = get_valid_values("physics", "reynolds_number")

        assert valid_values is None


class TestContextAndValidationIntegration:
    """Integration tests for context and validation together."""

    def test_full_workflow_with_validation(self):
        """Test full workflow: collect data and validate."""
        context = ConversationContext()
        context.initialize_case_info()

        # Collect physics data
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")

        # Validate physics
        physics_data = context.get_nested_field("collected_data.physics")
        result = validate_category("physics", physics_data)

        assert result.is_valid is True

    def test_context_with_question_history(self):
        """Test context maintaining question history during validation."""
        context = ConversationContext()
        context.initialize_case_info()

        # Add questions
        context.add_question_to_history("physics", "What flow type?", "incompressible")
        context.add_question_to_history("physics", "Steady or transient?", "steady")

        # Update data based on answers
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")

        # Get history and validate
        history = context.get_questions_by_topic("physics")
        assert len(history) == 2

        physics_data = context.get_nested_field("collected_data.physics")
        result = validate_category("physics", physics_data)
        assert result.is_valid is True
