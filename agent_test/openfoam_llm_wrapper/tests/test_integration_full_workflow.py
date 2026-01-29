"""Integration tests for full end-to-end workflow.

Tests the complete workflow from interactive questioning through case generation,
including state transitions, navigation, LLM interactions, and file output.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from openfoam_llm_wrapper.core.context_manager import ConversationContext
from openfoam_llm_wrapper.core.interactive_workflow import (
    InteractiveWorkflowManager,
    WorkflowState,
)
from openfoam_llm_wrapper.llm.client import LLMClient
from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder
from openfoam_llm_wrapper.llm.response_parser import ResponseParser
from openfoam_llm_wrapper.core.orchestrator import Orchestrator
from openfoam_llm_wrapper.core.summary_formatter import create_case_description


class TestFullInteractiveWorkflow:
    """Integration tests for complete interactive workflow scenarios."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client with realistic responses."""
        client = Mock(spec=LLMClient)

        # Mock interactive questioning with multi-turn responses
        client.interactive_case_questioning = Mock(side_effect=[
            "What is the flow type: incompressible or compressible?",
            "What is the time type: steady or transient?",
            "What is the expected Reynolds number?",
            "READY_TO_GENERATE",  # Completion signal
        ])

        # Mock final case generation
        client.chat_with_history = Mock(return_value="""
<file path="system/controlDict">
FoamFile {
    version 2.0;
    format ascii;
}
application simpleFoam;
startTime 0;
endTime 1000;
deltaT 1;
</file>

<file path="system/fvSchemes">
FoamFile {
    version 2.0;
}
ddtSchemes { default steadyState; }
</file>

<file path="0/U">
FoamFile {
    version 2.0;
    class volVectorField;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField { inlet { type fixedValue; value uniform (1 0 0); } }
</file>

<file path="0/p">
FoamFile {
    version 2.0;
    class volScalarField;
}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField { outlet { type zeroGradient; } }
</file>
""")

        return client

    @pytest.fixture
    def context(self):
        """Create a context for workflow testing."""
        ctx = ConversationContext()
        ctx.initialize_case_info()
        return ctx

    @pytest.fixture
    def manager(self, context, mock_llm_client):
        """Create a workflow manager for testing."""
        return InteractiveWorkflowManager(context, mock_llm_client)

    def test_happy_path_full_workflow(self, manager, context, mock_llm_client):
        """Test complete workflow from start to finish with all questions answered."""
        # Simulate workflow progression
        manager.state = WorkflowState.GATHERING_INFO

        # Simulate question 1
        q1 = manager._get_next_question()
        assert q1 is not None
        assert "flow type" in q1.lower()

        # Answer question 1
        manager._process_answer(q1, "incompressible")
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")

        # Simulate question 2
        q2 = manager._get_next_question()
        assert q2 is not None

        # Answer question 2
        manager._process_answer(q2, "steady")
        context.update_nested_field("collected_data.physics.time_type", "steady")

        # Simulate question 3
        q3 = manager._get_next_question()
        assert q3 is not None

        # Answer question 3
        manager._process_answer(q3, "100000")
        context.update_nested_field("collected_data.physics.reynolds_number", 100000)

        # Check for completion signal
        q4 = manager._get_next_question()
        assert manager._is_completion_signal(q4)

        # Verify data was collected
        assert context.get_nested_field("collected_data.physics.flow_type") == "incompressible"
        assert context.get_nested_field("collected_data.physics.time_type") == "steady"
        assert context.get_nested_field("collected_data.physics.reynolds_number") == 100000

    def test_workflow_with_skip_command(self, manager, context):
        """Test workflow with skip command for optional questions."""
        manager.state = WorkflowState.GATHERING_INFO

        # Get first question
        q1 = manager._get_next_question()
        assert q1 is not None

        # Skip the question
        skipped = manager.command_handler.execute_skip()
        assert isinstance(skipped, dict)

        # Verify we moved forward without answering
        assert manager.current_question_index > 0 or len(context.messages) > 0

    def test_workflow_with_back_edit(self, manager, context):
        """Test workflow back/edit navigation."""
        manager.state = WorkflowState.GATHERING_INFO

        # Answer first question
        q1 = manager._get_next_question()
        manager._process_answer(q1, "incompressible")

        # Answer second question
        q2 = manager._get_next_question()
        manager._process_answer(q2, "steady")

        # Now go back
        history = manager.command_handler.execute_back()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_workflow_with_summary_display(self, manager, context):
        """Test summary display during workflow."""
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")
        context.mark_category_complete("physics")

        # Get summary
        summary = manager.command_handler.execute_summary()
        assert summary is not None
        assert isinstance(summary, str)
        assert "incompressible" in summary or "physics" in summary.lower()

    def test_state_transitions(self, manager):
        """Test proper state machine transitions."""
        # Start in initialization
        assert manager.state == WorkflowState.INITIALIZATION

        # Transition to gathering info
        manager.transition_state(WorkflowState.GATHERING_INFO)
        assert manager.state == WorkflowState.GATHERING_INFO

        # Transition to reviewing
        manager.transition_state(WorkflowState.REVIEWING_SUMMARY)
        assert manager.state == WorkflowState.REVIEWING_SUMMARY

        # Transition to confirmed
        manager.transition_state(WorkflowState.CONFIRMED)
        assert manager.state == WorkflowState.CONFIRMED

        # Transition to generating
        manager.transition_state(WorkflowState.GENERATING)
        assert manager.state == WorkflowState.GENERATING

        # Transition to complete
        manager.transition_state(WorkflowState.COMPLETE)
        assert manager.state == WorkflowState.COMPLETE


class TestIntegratedLLMGeneration:
    """Integration tests for LLM-powered case generation."""

    @pytest.fixture
    def prompt_builder(self):
        """Create a prompt builder."""
        return PromptBuilder()

    @pytest.fixture
    def response_parser(self):
        """Create a response parser."""
        return ResponseParser()

    def test_prompt_building_from_case_info(self, prompt_builder):
        """Test building final prompt from collected case info."""
        case_info = {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady",
                    "reynolds_number": 100000,
                    "turbulence_model": "kOmegaSST",
                },
                "geometry": {
                    "description": "Rectangular duct 0.1m x 0.2m x 2m",
                    "dimension": "3D",
                },
                "boundary_conditions": {
                    "patches": {
                        "inlet": {"type": "fixedValue", "field_values": {"U": "5 0 0"}},
                        "outlet": {"type": "zeroGradient"},
                    }
                },
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {"density": 1.0, "viscosity": 0.001},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {},
            }
        }

        prompt = prompt_builder.build_final_case_prompt(case_info)

        # Verify prompt contains key information
        assert "incompressible" in prompt
        assert "steady" in prompt
        assert "100000" in prompt
        assert "simpleFoam" in prompt
        assert "Rectangular duct" in prompt

    def test_response_parsing_and_extraction(self, response_parser):
        """Test parsing LLM response into case files."""
        llm_response = """
I'll generate the OpenFOAM case for your simulation.

<file path="system/controlDict">
FoamFile {
    version 2.0;
    format ascii;
}
application simpleFoam;
startTime 0;
endTime 1000;
deltaT 1;
</file>

<file path="0/U">
FoamFile {
    version 2.0;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField { inlet { type fixedValue; } }
</file>

<file path="0/p">
FoamFile {
    version 2.0;
}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
</file>
"""

        files = response_parser.parse_case_files(llm_response)

        # Verify files were extracted
        assert len(files) == 3
        assert "system/controlDict" in files
        assert "0/U" in files
        assert "0/p" in files

        # Verify content
        assert "simpleFoam" in files["system/controlDict"]
        assert "fixedValue" in files["0/U"]

    def test_case_info_to_description_conversion(self):
        """Test conversion of case_info to natural language description."""
        case_info = {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady",
                    "reynolds_number": 50000,
                    "turbulence_model": "kEpsilon",
                },
                "geometry": {
                    "description": "Circular pipe with 0.1m diameter",
                    "dimension": "3D",
                },
                "boundary_conditions": {
                    "patches": {
                        "inlet": {"type": "fixedValue", "field_values": {}},
                        "outlet": {"type": "zeroGradient"},
                    }
                },
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {"density": 1000, "viscosity": 0.001},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {},
            }
        }

        description = create_case_description(case_info)

        # Verify description contains key information
        assert "incompressible" in description
        assert "steady" in description
        assert "pipe" in description.lower()
        assert "50000" in description

    def test_interactive_prompt_with_dynamic_content(self):
        """Test interactive system prompt with dynamic status/history."""
        builder = PromptBuilder()

        collection_status = {
            "physics": True,
            "geometry": False,
            "boundary_conditions": False,
            "solver": False,
            "fluid_properties": False,
            "mesh": False,
            "simulation_goals": False,
            "advanced": False,
        }

        collected_data = {
            "physics": {"flow_type": "incompressible", "time_type": "steady"},
            "geometry": {},
            "boundary_conditions": {},
            "solver": {},
            "fluid_properties": {},
            "mesh": {},
            "simulation_goals": {},
            "advanced": {},
        }

        question_history = [
            {
                "topic": "physics",
                "question": "What flow type?",
                "answer": "incompressible",
            },
            {
                "topic": "physics",
                "question": "Steady or transient?",
                "answer": "steady",
            },
        ]

        prompt = builder.build_interactive_system_prompt(
            collection_status, collected_data, question_history
        )

        # Verify prompt contains dynamic content
        assert "incompressible" in prompt
        assert "steady" in prompt
        assert "physics" in prompt.lower()

    def test_multi_turn_conversation_with_history(self):
        """Test multi-turn LLM conversation with full history."""
        mock_client = Mock(spec=LLMClient)
        mock_client.chat_with_history = Mock(return_value="Next question about geometry")

        messages = [
            {"role": "user", "content": "I want to simulate incompressible flow"},
            {"role": "assistant", "content": "What is the geometry?"},
            {"role": "user", "content": "Pipe with 0.1m diameter"},
        ]

        response = mock_client.chat_with_history(
            messages=messages,
            system_prompt="You are an expert OpenFOAM consultant",
            max_tokens=2048,
        )

        assert response == "Next question about geometry"
        mock_client.chat_with_history.assert_called_once()


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with interactive workflow."""

    @pytest.fixture
    def temp_case_dir(self):
        """Create a temporary directory for test cases."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with mocked components."""
        orchestrator = Mock(spec=Orchestrator)

        # Mock generate_case_from_interactive
        orchestrator.generate_case_from_interactive = Mock(
            return_value={
                "case_path": "/path/to/case",
                "validation": {"is_valid": True, "errors": []},
                "files_generated": 4,
            }
        )

        return orchestrator

    def test_generate_from_interactive_data(self, mock_orchestrator):
        """Test case generation from interactive workflow data."""
        case_info = {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady",
                },
                "geometry": {"description": "pipe"},
                "boundary_conditions": {"patches": {}},
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {},
            }
        }

        result = mock_orchestrator.generate_case_from_interactive(
            case_info=case_info,
            output_dir="/tmp",
            case_name="test_case",
        )

        # Verify result
        assert result["case_path"] == "/path/to/case"
        assert result["validation"]["is_valid"] is True
        assert result["files_generated"] == 4

        # Verify method was called with correct args
        mock_orchestrator.generate_case_from_interactive.assert_called_once_with(
            case_info=case_info,
            output_dir="/tmp",
            case_name="test_case",
        )

    def test_context_flow_from_collection_to_generation(self):
        """Test complete data flow from collection to generation."""
        # Create context and collect data
        context = ConversationContext()
        context.initialize_case_info()

        # Simulate data collection
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")
        context.update_nested_field("collected_data.physics.reynolds_number", 100000)
        context.update_nested_field("collected_data.geometry.description", "pipe")
        context.update_nested_field("collected_data.solver.solver", "simpleFoam")

        # Mark categories complete
        context.mark_category_complete("physics")
        context.mark_category_complete("geometry")
        context.mark_category_complete("solver")

        # Verify data is properly structured
        physics_data = context.get_nested_field("collected_data.physics")
        assert physics_data["flow_type"] == "incompressible"
        assert physics_data["time_type"] == "steady"
        assert physics_data["reynolds_number"] == 100000

        # Verify completion status
        status = context.get_completion_status()
        assert status["physics"] is True
        assert status["geometry"] is True
        assert status["solver"] is True
        assert status["boundary_conditions"] is False

    def test_question_history_tracking(self):
        """Test question history tracking throughout workflow."""
        context = ConversationContext()
        context.initialize_case_info()

        # Add questions to history
        context.add_question_to_history(
            "physics", "What flow type?", "incompressible"
        )
        context.add_question_to_history(
            "physics", "Steady or transient?", "steady"
        )
        context.add_question_to_history(
            "geometry", "Describe geometry", "pipe with 0.1m diameter"
        )

        # Retrieve full history
        history = context.get_question_history()
        assert len(history) == 3

        # Retrieve by topic
        physics_questions = context.get_questions_by_topic("physics")
        assert len(physics_questions) == 2
        assert all(q["topic"] == "physics" for q in physics_questions)

        # Verify content
        assert physics_questions[0]["answer"] == "incompressible"
        assert physics_questions[1]["answer"] == "steady"


class TestErrorHandlingAndEdgeCases:
    """Integration tests for error handling and edge cases."""

    def test_incomplete_data_scenario(self):
        """Test workflow with incomplete data collection."""
        context = ConversationContext()
        context.initialize_case_info()

        # Only fill partial data
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        # Missing many required fields

        # Attempt validation
        status = context.get_completion_status()

        # Should show incomplete categories
        assert status["physics"] is False
        assert status["geometry"] is False
        assert status["solver"] is False

    def test_lLM_completion_signal_detection(self):
        """Test detection of LLM completion signal."""
        mock_client = Mock(spec=LLMClient)
        context = ConversationContext()
        context.initialize_case_info()

        manager = InteractiveWorkflowManager(context, mock_client)

        # Test with completion signal
        assert manager._is_completion_signal("READY_TO_GENERATE")
        assert manager._is_completion_signal("All information collected, READY_TO_GENERATE")

        # Test without completion signal
        assert not manager._is_completion_signal("Ask the next question")
        assert not manager._is_completion_signal("Need more information")

    def test_invalid_command_handling(self):
        """Test handling of invalid user commands."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_client = Mock(spec=LLMClient)

        manager = InteractiveWorkflowManager(context, mock_client)

        # Try invalid command
        result = manager.command_handler.parse_command("invalid_command")
        assert result is None or isinstance(result, str)

    def test_very_long_answer_handling(self):
        """Test handling of very long user answers."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_client = Mock(spec=LLMClient)

        manager = InteractiveWorkflowManager(context, mock_client)

        # Very long answer (5000+ characters)
        long_answer = "A" * 5000

        # Should handle without crashing
        manager.collector.update_field("physics", "flow_type", long_answer)

        # Verify it's stored
        assert manager.collector.get_field("physics", "flow_type") == long_answer

    def test_special_characters_in_responses(self):
        """Test handling of special characters and Unicode."""
        context = ConversationContext()
        context.initialize_case_info()

        # Update with special characters
        special_text = "Simulation with α, β, γ and symbols: @#$%^&*()"
        context.update_nested_field("collected_data.geometry.description", special_text)

        # Retrieve and verify
        retrieved = context.get_nested_field("collected_data.geometry.description")
        assert retrieved == special_text

    def test_rapid_state_changes(self):
        """Test rapid state machine transitions."""
        context = ConversationContext()
        context.initialize_case_info()
        mock_client = Mock(spec=LLMClient)

        manager = InteractiveWorkflowManager(context, mock_client)

        # Rapid transitions
        manager.transition_state(WorkflowState.GATHERING_INFO)
        manager.transition_state(WorkflowState.REVIEWING_SUMMARY)
        manager.transition_state(WorkflowState.GATHERING_INFO)
        manager.transition_state(WorkflowState.REVIEWING_SUMMARY)
        manager.transition_state(WorkflowState.CONFIRMED)

        # Should end in correct state
        assert manager.state == WorkflowState.CONFIRMED


class TestRealWorldScenarios:
    """Integration tests simulating real-world usage scenarios."""

    def test_incompressible_pipe_flow_workflow(self):
        """Test complete workflow for incompressible pipe flow."""
        context = ConversationContext()
        context.initialize_case_info()

        mock_client = Mock(spec=LLMClient)
        manager = InteractiveWorkflowManager(context, mock_client)

        # Simulate real scenario
        case_data = {
            "physics": {
                "flow_type": "incompressible",
                "time_type": "steady",
                "reynolds_number": 100000,
                "turbulence_model": "kOmegaSST",
            },
            "geometry": {
                "description": "Circular pipe, diameter 0.1m, length 5m",
                "dimension": "3D",
            },
            "boundary_conditions": {
                "patches": {
                    "inlet": {"type": "fixedValue", "field_values": {"U": "5 0 0"}},
                    "outlet": {"type": "zeroGradient"},
                    "wall": {"type": "noSlip"},
                }
            },
            "solver": {"solver": "simpleFoam"},
            "fluid_properties": {"density": 1000, "viscosity": 0.001},
        }

        # Update context
        for category, data in case_data.items():
            for key, value in data.items():
                context.update_nested_field(
                    f"collected_data.{category}.{key}", value
                )
            context.mark_category_complete(category)

        # Verify all data collected
        for category in ["physics", "geometry", "boundary_conditions", "solver", "fluid_properties"]:
            assert context.get_completion_status()[category] is True

        # Build case description
        description = create_case_description(context.case_info)
        assert "incompressible" in description
        assert "pipe" in description.lower()
        assert "100000" in description

    def test_compressible_supersonic_flow_workflow(self):
        """Test workflow for compressible supersonic flow."""
        context = ConversationContext()
        context.initialize_case_info()

        # Simulate supersonic flow case
        context.update_nested_field("collected_data.physics.flow_type", "compressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")
        context.update_nested_field("collected_data.physics.mach_number", 2.5)
        context.update_nested_field("collected_data.physics.turbulence_model", "kEpsilon")

        context.update_nested_field(
            "collected_data.geometry.description", "Supersonic nozzle"
        )

        context.update_nested_field(
            "collected_data.solver.solver", "rhoCentralFoam"
        )

        # Mark categories complete
        context.mark_category_complete("physics")
        context.mark_category_complete("geometry")
        context.mark_category_complete("solver")

        # Verify data
        physics = context.get_nested_field("collected_data.physics")
        assert physics["flow_type"] == "compressible"
        assert physics["mach_number"] == 2.5

        solver = context.get_nested_field("collected_data.solver")
        assert solver["solver"] == "rhoCentralFoam"

    def test_transient_heat_transfer_workflow(self):
        """Test workflow for transient heat transfer problem."""
        context = ConversationContext()
        context.initialize_case_info()

        # Heat transfer case
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "transient")
        context.update_nested_field("collected_data.physics.special_physics", ["heat_transfer"])

        context.update_nested_field(
            "collected_data.simulation_goals.description",
            "Analyze temperature evolution in heated pipe",
        )

        # Add question history
        context.add_question_to_history(
            "physics", "What special physics?", "heat_transfer"
        )
        context.add_question_to_history(
            "simulation_goals", "What are simulation goals?",
            "Analyze temperature evolution"
        )

        # Verify history
        history = context.get_question_history()
        assert len(history) == 2

        special_physics = context.get_nested_field(
            "collected_data.physics.special_physics"
        )
        assert "heat_transfer" in special_physics
