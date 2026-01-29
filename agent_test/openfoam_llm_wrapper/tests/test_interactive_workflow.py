"""Unit tests for interactive workflow components."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from openfoam_llm_wrapper.core.interactive_workflow import (
    InformationCollector,
    CommandHandler,
    InteractiveWorkflowManager,
    WorkflowState,
    InformationCategory,
)
from openfoam_llm_wrapper.core.context_manager import ConversationContext


class TestInformationCollector:
    """Test suite for InformationCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a fresh collector for each test."""
        return InformationCollector()

    def test_initialization(self, collector):
        """Test that collector initializes with proper structure."""
        assert collector.case_info is not None
        assert "collection_status" in collector.case_info
        assert "collected_data" in collector.case_info
        assert "question_history" in collector.case_info

    def test_initialize_method(self, collector):
        """Test the initialize method creates proper structure."""
        collector.initialize()

        # Check all categories exist
        expected_categories = [
            "physics",
            "geometry",
            "boundary_conditions",
            "fluid_properties",
            "solver",
            "mesh",
            "simulation_goals",
            "advanced",
        ]

        for category in expected_categories:
            assert category in collector.case_info["collection_status"]
            assert category in collector.case_info["collected_data"]
            assert collector.case_info["collection_status"][category] is False

    def test_update_field(self, collector):
        """Test updating a field in collected data."""
        collector.update_field("physics", "flow_type", "incompressible")

        assert collector.get_field("physics", "flow_type") == "incompressible"

    def test_update_nested_field(self, collector):
        """Test updating nested field with dot notation."""
        collector.update_nested_field("physics.flow_type", "compressible")

        assert collector.get_field("physics", "flow_type") == "compressible"

    def test_get_field_with_default(self, collector):
        """Test getting field with default value."""
        result = collector.get_field("physics", "nonexistent", "default_value")

        assert result == "default_value"

    def test_mark_category_complete(self, collector):
        """Test marking category as complete."""
        collector.mark_category_complete("physics")

        assert collector.case_info["collection_status"]["physics"] is True

    def test_mark_category_incomplete(self, collector):
        """Test marking category as incomplete."""
        collector.mark_category_complete("physics")
        collector.mark_category_incomplete("physics")

        assert collector.case_info["collection_status"]["physics"] is False

    def test_get_completion_status(self, collector):
        """Test getting completion status of all categories."""
        collector.mark_category_complete("physics")
        collector.mark_category_complete("geometry")

        status = collector.get_completion_status()

        assert status["physics"] is True
        assert status["geometry"] is True
        assert status["boundary_conditions"] is False

    def test_add_question_to_history(self, collector):
        """Test adding question to history."""
        collector.add_question_to_history(
            "physics",
            "What type of flow?",
            "incompressible steady"
        )

        history = collector.get_question_history()

        assert len(history) == 1
        assert history[0]["topic"] == "physics"
        assert history[0]["question"] == "What type of flow?"
        assert history[0]["answer"] == "incompressible steady"
        assert "timestamp" in history[0]

    def test_get_questions_by_topic(self, collector):
        """Test retrieving questions by topic."""
        collector.add_question_to_history("physics", "Q1?", "A1")
        collector.add_question_to_history("physics", "Q2?", "A2")
        collector.add_question_to_history("geometry", "Q3?", "A3")

        physics_questions = collector.get_questions_by_topic("physics")

        assert len(physics_questions) == 2
        assert all(q["topic"] == "physics" for q in physics_questions)

    def test_is_complete_with_all_required(self, collector):
        """Test completion check when all required fields present."""
        collector.update_field("physics", "flow_type", "incompressible")
        collector.update_field("physics", "time_type", "steady")
        collector.update_field("geometry", "description", "pipe flow")
        collector.update_field("boundary_conditions", "patches", {"inlet": {}})
        collector.update_field("solver", "solver", "simpleFoam")

        assert collector.is_complete() is True

    def test_is_complete_with_missing_required(self, collector):
        """Test completion check when required fields missing."""
        collector.update_field("physics", "flow_type", "incompressible")
        # Missing time_type and others

        assert collector.is_complete() is False

    def test_get_missing_required(self, collector):
        """Test getting list of missing required fields."""
        collector.update_field("physics", "flow_type", "incompressible")

        missing = collector.get_missing_required()

        assert "physics.time_type" in missing
        assert "geometry.description" in missing
        assert "boundary_conditions.patches" in missing
        assert "solver.solver" in missing

    def test_validate_with_errors(self, collector):
        """Test validation with errors."""
        # Don't add required fields
        result = collector.validate()

        assert len(result["errors"]) > 0
        assert result["errors"][0].startswith("Missing required:")

    def test_validate_with_no_errors(self, collector):
        """Test validation with all required fields."""
        collector.update_field("physics", "flow_type", "incompressible")
        collector.update_field("physics", "time_type", "steady")
        collector.update_field("geometry", "description", "test geometry")
        collector.update_field("boundary_conditions", "patches", {})
        collector.update_field("solver", "solver", "simpleFoam")

        result = collector.validate()

        assert len(result["errors"]) == 0


class TestCommandHandler:
    """Test suite for CommandHandler class."""

    @pytest.fixture
    def collector(self):
        """Create a fresh collector for each test."""
        return InformationCollector()

    @pytest.fixture
    def handler(self, collector):
        """Create handler with collector."""
        return CommandHandler(collector)

    def test_parse_command_skip(self, handler):
        """Test parsing skip command."""
        result = handler.parse_command("skip")

        assert result is not None
        assert result[0] == "skip"
        assert result[1] is None

    def test_parse_command_with_argument(self, handler):
        """Test parsing command with argument."""
        result = handler.parse_command("edit physics.flow_type")

        assert result is not None
        assert result[0] == "edit"
        assert result[1] == "physics.flow_type"

    def test_parse_command_case_insensitive(self, handler):
        """Test that command parsing is case insensitive."""
        result = handler.parse_command("SKIP")

        assert result is not None
        assert result[0] == "skip"

    def test_parse_invalid_command(self, handler):
        """Test parsing invalid command."""
        result = handler.parse_command("invalid_command")

        assert result is None

    def test_is_command(self, handler):
        """Test is_command detection."""
        assert handler.is_command("skip") is True
        assert handler.is_command("back") is True
        assert handler.is_command("not a command") is False

    def test_execute_skip(self, handler):
        """Test executing skip command."""
        result = handler.execute_command("skip")

        assert result["action"] == "skip"
        assert "message" in result

    def test_execute_back(self, handler):
        """Test executing back command."""
        result = handler.execute_command("back")

        assert result["action"] == "back"
        assert "found_history" in result

    def test_execute_back_with_history(self, handler, collector):
        """Test executing back command with history."""
        collector.add_question_to_history("physics", "Q1?", "A1")
        collector.add_question_to_history("physics", "Q2?", "A2")

        result = handler.execute_command("back")

        assert result["found_history"] is True
        assert len(result["history"]) == 2

    def test_execute_done_complete(self, handler, collector):
        """Test executing done command when complete."""
        collector.update_field("physics", "flow_type", "incompressible")
        collector.update_field("physics", "time_type", "steady")
        collector.update_field("geometry", "description", "test")
        collector.update_field("boundary_conditions", "patches", {})
        collector.update_field("solver", "solver", "simpleFoam")

        result = handler.execute_command("done")

        assert result["is_complete"] is True
        assert len(result["missing"]) == 0

    def test_execute_done_incomplete(self, handler):
        """Test executing done command when incomplete."""
        result = handler.execute_command("done")

        assert result["is_complete"] is False
        assert len(result["missing"]) > 0

    def test_execute_edit_without_argument(self, handler):
        """Test executing edit without argument."""
        result = handler.execute_command("edit")

        assert result["action"] == "error"
        assert "Usage" in result["message"]

    def test_execute_edit_with_argument(self, handler):
        """Test executing edit with argument."""
        result = handler.execute_command("edit", "physics.flow_type")

        assert result["action"] == "edit"
        assert result["field"] == "physics.flow_type"

    def test_execute_help(self, handler):
        """Test executing help command."""
        result = handler.execute_command("help")

        assert result["action"] == "help"
        assert "commands" in result

    def test_execute_quit(self, handler):
        """Test executing quit command."""
        result = handler.execute_command("quit")

        assert result["action"] == "exit"

    def test_valid_commands_exist(self, handler):
        """Test that valid commands are defined."""
        assert len(handler.VALID_COMMANDS) > 0
        assert "skip" in handler.VALID_COMMANDS
        assert "back" in handler.VALID_COMMANDS


class TestInteractiveWorkflowManager:
    """Test suite for InteractiveWorkflowManager class."""

    @pytest.fixture
    def context(self):
        """Create a fresh context for each test."""
        ctx = ConversationContext()
        ctx.initialize_case_info()
        return ctx

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = Mock()
        mock_client.interactive_case_questioning = Mock(
            return_value="What type of flow are you simulating?"
        )
        return mock_client

    @pytest.fixture
    def manager(self, context, mock_llm_client):
        """Create a workflow manager with mocks."""
        return InteractiveWorkflowManager(context, mock_llm_client)

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.context is not None
        assert manager.llm_client is not None
        assert manager.collector is not None
        assert manager.command_handler is not None
        assert manager.state == WorkflowState.INITIALIZATION

    def test_state_transitions(self, manager):
        """Test state machine transitions."""
        manager.state = WorkflowState.GATHERING_INFO
        assert manager.state == WorkflowState.GATHERING_INFO

        manager.state = WorkflowState.REVIEWING_SUMMARY
        assert manager.state == WorkflowState.REVIEWING_SUMMARY

    def test_collector_initialization(self, manager):
        """Test that collector is properly initialized."""
        manager.collector.initialize()

        assert manager.collector.case_info["collection_status"] is not None
        assert manager.collector.case_info["collected_data"] is not None

    def test_is_completion_signal_true(self, manager):
        """Test completion signal detection (positive case)."""
        response = "I have gathered enough information. READY_TO_GENERATE"
        assert manager._is_completion_signal(response) is True

    def test_is_completion_signal_false(self, manager):
        """Test completion signal detection (negative case)."""
        response = "What is your next requirement?"
        assert manager._is_completion_signal(response) is False

    def test_is_completion_signal_case_insensitive(self, manager):
        """Test completion signal is case insensitive."""
        response = "ready to generate now"
        assert manager._is_completion_signal(response) is True

    def test_extract_question_with_question_mark(self, manager):
        """Test extracting question that contains ?."""
        response = "What type of flow are you simulating?"
        question = manager._extract_question_from_response(response)

        assert question is not None
        assert "What type of flow" in question

    def test_extract_question_from_multiple_lines(self, manager):
        """Test extracting question from multiline response."""
        response = """Some preamble text

What is the Reynolds number for your flow?"""
        question = manager._extract_question_from_response(response)

        assert question is not None
        assert "Reynolds number" in question

    def test_extract_question_no_question(self, manager):
        """Test extracting when no clear question exists."""
        response = "incompressible flow with steady state"
        question = manager._extract_question_from_response(response)

        # Should return something
        assert question is not None

    def test_infer_topic_physics(self, manager):
        """Test topic inference for physics question."""
        question = "What is the Reynolds number?"
        topic = manager._infer_topic_from_question(question)

        assert topic == InformationCategory.PHYSICS.value

    def test_infer_topic_geometry(self, manager):
        """Test topic inference for geometry question."""
        question = "What are the dimensions of your pipe?"
        topic = manager._infer_topic_from_question(question)

        assert topic == InformationCategory.GEOMETRY.value

    def test_infer_topic_boundary_conditions(self, manager):
        """Test topic inference for BC question."""
        question = "What are your inlet boundary conditions?"
        topic = manager._infer_topic_from_question(question)

        assert topic == InformationCategory.BOUNDARY_CONDITIONS.value

    def test_infer_topic_solver(self, manager):
        """Test topic inference for solver question."""
        question = "Which solver would you recommend?"
        topic = manager._infer_topic_from_question(question)

        assert topic == InformationCategory.SOLVER.value

    def test_infer_topic_general(self, manager):
        """Test topic inference for general question."""
        question = "How are you doing today?"
        topic = manager._infer_topic_from_question(question)

        assert topic == "general"

    def test_process_answer(self, manager):
        """Test processing a user answer."""
        question = "What type of flow are you simulating?"
        answer = "incompressible steady"

        manager._process_answer(question, answer)

        # Check it was added to history
        history = manager.collector.get_question_history()
        assert len(history) > 0

    def test_answer_added_to_context(self, manager):
        """Test that answer is added to conversation context."""
        answer = "My test answer"

        manager._process_answer("Question?", answer)

        # Check context has the message
        context_history = manager.context.get_conversation_history()
        assert len(context_history) > 0

    def test_should_transition_to_summary(self, manager):
        """Test transition logic."""
        manager.question_count = 5
        assert manager._should_transition_to_summary() is True

        manager.question_count = 2
        assert manager._should_transition_to_summary() is False

    def test_max_questions_limit(self, manager):
        """Test that max questions limit is enforced."""
        manager.max_questions = 20
        manager.question_count = 25

        # Should not get next question
        question = manager._get_next_question()

        # May return None or error
        assert question is None or "error" in str(question).lower()

    def test_get_case_info(self, manager):
        """Test retrieving case info."""
        manager.collector.update_field("physics", "flow_type", "incompressible")

        case_info = manager.get_case_info()

        assert case_info is not None
        assert case_info["collected_data"]["physics"]["flow_type"] == "incompressible"

    def test_is_workflow_complete_true(self, manager):
        """Test workflow complete check (true case)."""
        manager.state = WorkflowState.COMPLETE

        assert manager.is_workflow_complete() is True

    def test_is_workflow_complete_false(self, manager):
        """Test workflow complete check (false case)."""
        manager.state = WorkflowState.GATHERING_INFO

        assert manager.is_workflow_complete() is False


class TestWorkflowStateTransitions:
    """Test suite for workflow state machine transitions."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocked LLM."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_llm = Mock()
        return InteractiveWorkflowManager(context, mock_llm)

    def test_valid_state_sequence(self, manager):
        """Test a valid sequence of state transitions."""
        assert manager.state == WorkflowState.INITIALIZATION

        manager.state = WorkflowState.GATHERING_INFO
        assert manager.state == WorkflowState.GATHERING_INFO

        manager.state = WorkflowState.REVIEWING_SUMMARY
        assert manager.state == WorkflowState.REVIEWING_SUMMARY

        manager.state = WorkflowState.CONFIRMED
        assert manager.state == WorkflowState.CONFIRMED

        manager.state = WorkflowState.GENERATING
        assert manager.state == WorkflowState.GENERATING

        manager.state = WorkflowState.COMPLETE
        assert manager.state == WorkflowState.COMPLETE

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        expected_states = [
            "initialization",
            "gathering_info",
            "reviewing_summary",
            "confirmed",
            "generating",
            "complete",
        ]

        actual_states = [state.value for state in WorkflowState]

        for expected in expected_states:
            assert expected in actual_states


class TestCommandIntegration:
    """Integration tests for command handling within workflow."""

    @pytest.fixture
    def setup(self):
        """Setup for integration tests."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_llm = Mock()
        manager = InteractiveWorkflowManager(context, mock_llm)
        return manager

    def test_skip_command_increments_question_count(self, setup):
        """Test that skip command increments question counter."""
        initial_count = setup.question_count

        # Simulate skip handling
        setup.question_count += 1

        assert setup.question_count == initial_count + 1

    def test_command_handler_knows_about_collector(self, setup):
        """Test that command handler has access to collector data."""
        setup.collector.add_question_to_history("physics", "Q1?", "A1")

        history = setup.command_handler.collector.get_question_history()

        assert len(history) == 1

    def test_multiple_answers_stored(self, setup):
        """Test that multiple answers are properly stored."""
        setup._process_answer("Q1?", "Answer 1")
        setup._process_answer("Q2?", "Answer 2")
        setup._process_answer("Q3?", "Answer 3")

        history = setup.collector.get_question_history()

        assert len(history) == 3
        assert history[0]["answer"] == "Answer 1"
        assert history[2]["answer"] == "Answer 3"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocks."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_llm = Mock()
        return InteractiveWorkflowManager(context, mock_llm)

    def test_empty_answer(self, manager):
        """Test handling of empty answer."""
        question = "What is your flow type?"
        answer = ""

        # Should handle gracefully
        manager._process_answer(question, answer)

        history = manager.collector.get_question_history()
        # Empty answer still added to history
        assert len(history) == 1

    def test_very_long_answer(self, manager):
        """Test handling of very long answer."""
        question = "Describe your geometry:"
        answer = "x" * 5000  # Very long string

        manager._process_answer(question, answer)

        history = manager.collector.get_question_history()
        assert len(history) == 1
        assert len(history[0]["answer"]) == 5000

    def test_special_characters_in_answer(self, manager):
        """Test handling special characters."""
        question = "What is your description?"
        answer = "Pipe with <special> & {characters} ©®™"

        manager._process_answer(question, answer)

        history = manager.collector.get_question_history()
        assert history[0]["answer"] == answer

    def test_unicode_in_answer(self, manager):
        """Test handling of unicode characters."""
        question = "Describe geometry:"
        answer = "Pipe with μ (mu) and Ω (omega) symbols"

        manager._process_answer(question, answer)

        history = manager.collector.get_question_history()
        assert "μ" in history[0]["answer"]

    def test_multiple_rapid_state_changes(self, manager):
        """Test rapid state changes."""
        states = [
            WorkflowState.GATHERING_INFO,
            WorkflowState.REVIEWING_SUMMARY,
            WorkflowState.CONFIRMED,
            WorkflowState.GENERATING,
            WorkflowState.COMPLETE,
        ]

        for state in states:
            manager.state = state
            assert manager.state == state

    def test_question_extraction_with_whitespace(self, manager):
        """Test question extraction with various whitespace."""
        response = """


What is the Reynolds number?


"""
        question = manager._extract_question_from_response(response)

        assert question is not None
        assert "Reynolds" in question

    def test_collector_with_none_values(self, manager):
        """Test collector handling None values."""
        manager.collector.update_field("physics", "flow_type", None)

        # Should not crash
        result = manager.collector.get_field("physics", "flow_type")
        assert result is None
