"""Integration tests for case generation pipeline.

Tests the complete end-to-end case generation process from case_info through
file creation and validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder
from openfoam_llm_wrapper.llm.response_parser import ResponseParser
from openfoam_llm_wrapper.core.context_manager import ConversationContext
from openfoam_llm_wrapper.core.summary_formatter import create_case_description


class TestCaseGenerationPipeline:
    """Integration tests for the complete case generation pipeline."""

    @pytest.fixture
    def complete_case_info(self):
        """Create a complete case_info with all required fields."""
        return {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady",
                    "reynolds_number": 100000,
                    "turbulence_model": "kOmegaSST",
                    "special_physics": [],
                },
                "geometry": {
                    "description": "Rectangular duct 0.1m x 0.2m x 2m with uniform mesh",
                    "dimension": "3D",
                },
                "boundary_conditions": {
                    "patches": {
                        "inlet": {
                            "type": "fixedValue",
                            "field_values": {"U": "5 0 0"},
                        },
                        "outlet": {
                            "type": "zeroGradient",
                        },
                        "walls": {
                            "type": "noSlip",
                        },
                    }
                },
                "solver": {
                    "solver": "simpleFoam",
                    "algorithm": "SIMPLE",
                },
                "fluid_properties": {
                    "density": 1.0,
                    "viscosity": 0.001,
                },
                "mesh": {
                    "element_size": 0.01,
                    "refinement_regions": [],
                },
                "simulation_goals": {
                    "description": "Calculate velocity and pressure distribution",
                    "output_interval": 100,
                },
                "advanced": {
                    "residuals_threshold": 1e-6,
                },
            },
            "collection_status": {
                "physics": True,
                "geometry": True,
                "boundary_conditions": True,
                "solver": True,
                "fluid_properties": True,
                "mesh": True,
                "simulation_goals": True,
                "advanced": True,
            },
        }

    def test_case_description_generation(self, complete_case_info):
        """Test conversion of case_info to case description."""
        description = create_case_description(complete_case_info)

        # Verify key information is included
        assert description is not None
        assert len(description) > 0
        assert "incompressible" in description
        assert "steady" in description
        assert "100000" in description
        assert "simpleFoam" in description
        assert "Rectangular duct" in description

    def test_prompt_building_from_complete_case(self, complete_case_info):
        """Test building final generation prompt from complete case info."""
        builder = PromptBuilder()

        prompt = builder.build_final_case_prompt(complete_case_info)

        # Verify prompt structure
        assert prompt is not None
        assert len(prompt) > 500  # Should be substantial

        # Verify content
        assert "simpleFoam" in prompt
        assert "incompressible" in prompt
        assert "steady" in prompt
        assert "<file path=" in prompt

    def test_llm_response_to_files_conversion(self):
        """Test converting LLM response into structured files."""
        parser = ResponseParser()

        # Realistic LLM response with multiple files
        llm_response = """
I've generated a complete OpenFOAM case for your incompressible pipe flow simulation.

<file path="system/controlDict">
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     simpleFoam;
startTime       0;
endTime         1000;
deltaT          1;
writeInterval   100;

libs
(
    "libfvMomentumTransport.so"
);
</file>

<file path="system/fvSchemes">
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss QUICK;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
</file>

<file path="system/fvSolution">
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
</file>

<file path="0/U">
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (5 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (5 0 0);
    }

    outlet
    {
        type            zeroGradient;
    }

    walls
    {
        type            noSlip;
    }
}
</file>

<file path="0/p">
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    walls
    {
        type            zeroGradient;
    }
}
</file>

<file path="0/k">
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.1;

boundaryField
{
    inlet
    {
        type            turbulentIntensityKineticEnergyInlet;
        intensity       0.05;
        value           uniform 0.1;
    }

    outlet
    {
        type            zeroGradient;
    }

    walls
    {
        type            kqRWallFunction;
        value           uniform 0.1;
    }
}
</file>

<file path="0/omega">
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 1;

boundaryField
{
    inlet
    {
        type            turbulentMixingLengthFrequencyInlet;
        mixingLength    0.005;
        value           uniform 1;
    }

    outlet
    {
        type            zeroGradient;
    }

    walls
    {
        type            omegaWallFunction;
        value           uniform 1;
    }
}
</file>
"""

        files = parser.parse_case_files(llm_response)

        # Verify all files extracted
        assert len(files) == 7
        assert "system/controlDict" in files
        assert "system/fvSchemes" in files
        assert "system/fvSolution" in files
        assert "0/U" in files
        assert "0/p" in files
        assert "0/k" in files
        assert "0/omega" in files

        # Verify file contents
        assert "simpleFoam" in files["system/controlDict"]
        assert "ddtSchemes" in files["system/fvSchemes"]
        assert "GAMG" in files["system/fvSolution"]
        assert "fixedValue" in files["0/U"]
        assert "zeroGradient" in files["0/p"]

    def test_files_have_valid_structure(self):
        """Test that generated files have valid OpenFOAM structure."""
        parser = ResponseParser()

        response = """
<file path="0/U">
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (1 0 0);

boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type zeroGradient; }
}
</file>
"""

        files = parser.parse_case_files(response)
        content = files["0/U"]

        # Verify OpenFOAM structure
        assert "FoamFile" in content
        assert "version" in content
        assert "volVectorField" in content
        assert "dimensions" in content
        assert "internalField" in content
        assert "boundaryField" in content

    def test_case_generation_end_to_end(self, complete_case_info):
        """Test complete end-to-end case generation process."""
        builder = PromptBuilder()
        parser = ResponseParser()

        # Step 1: Build prompt from case info
        prompt = builder.build_final_case_prompt(complete_case_info)
        assert prompt is not None

        # Step 2: Simulate LLM response (in real scenario, call LLM)
        mock_llm_response = """
<file path="system/controlDict">
application simpleFoam;
startTime 0;
endTime 1000;
deltaT 1;
</file>

<file path="0/U">
dimensions [0 1 -1 0 0 0 0];
internalField uniform (5 0 0);
</file>

<file path="0/p">
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
</file>
"""

        # Step 3: Parse response into files
        files = parser.parse_case_files(mock_llm_response)

        # Step 4: Verify files
        assert len(files) == 3
        assert all(isinstance(content, str) for content in files.values())

        # Step 5: Verify key content
        assert "simpleFoam" in files["system/controlDict"]
        assert "uniform (5 0 0)" in files["0/U"]


class TestMultiTurnConversationIntegration:
    """Integration tests for multi-turn conversation flows."""

    def test_question_answering_sequence(self):
        """Test realistic question-answer sequence."""
        context = ConversationContext()
        context.initialize_case_info()

        # Simulate multi-turn conversation
        questions_answers = [
            ("What type of flow?", "incompressible"),
            ("Steady or transient?", "steady"),
            ("Expected Reynolds number?", "100000"),
            ("Turbulence model?", "k-omega SST"),
            ("Describe the geometry", "Pipe, diameter 0.1m, length 5m"),
            ("What are the inlet conditions?", "Fixed velocity 5 m/s in x"),
            ("What solver to use?", "simpleFoam"),
        ]

        # Process each Q&A
        for i, (question, answer) in enumerate(questions_answers):
            # Extract category from question
            if "flow" in question.lower():
                category = "physics"
                field = "flow_type"
            elif "steady" in question.lower():
                category = "physics"
                field = "time_type"
            elif "reynolds" in question.lower():
                category = "physics"
                field = "reynolds_number"
            elif "turbulence" in question.lower():
                category = "physics"
                field = "turbulence_model"
            elif "geometry" in question.lower():
                category = "geometry"
                field = "description"
            elif "inlet" in question.lower():
                category = "boundary_conditions"
                field = "inlet_type"
            elif "solver" in question.lower():
                category = "solver"
                field = "solver"
            else:
                continue

            # Record in history
            context.add_question_to_history(category, question, answer)

            # Update collected data
            if category == "physics" and field == "reynolds_number":
                context.update_nested_field(
                    f"collected_data.{category}.{field}", int(answer)
                )
            else:
                context.update_nested_field(
                    f"collected_data.{category}.{field}", answer
                )

        # Verify history
        history = context.get_question_history()
        assert len(history) >= 5

        # Verify data collection
        assert context.get_nested_field("collected_data.physics.flow_type") == "incompressible"
        assert context.get_nested_field("collected_data.physics.time_type") == "steady"
        assert context.get_nested_field("collected_data.physics.reynolds_number") == 100000

    def test_conversation_history_formatting(self):
        """Test formatting of conversation history for LLM."""
        context = ConversationContext()
        context.initialize_case_info()

        # Add questions to history
        context.add_question_to_history("physics", "What flow type?", "incompressible")
        context.add_question_to_history("physics", "Steady or transient?", "steady")
        context.add_question_to_history("geometry", "Describe geometry", "pipe")

        # Get questions
        history = context.get_question_history()

        # Verify each entry has required fields
        for entry in history:
            assert "topic" in entry
            assert "question" in entry
            assert "answer" in entry
            assert isinstance(entry, dict)

    def test_context_preservation_across_interactions(self):
        """Test that context is properly preserved across interactions."""
        context = ConversationContext()
        context.initialize_case_info()

        # First interaction
        context.add_message("user", "I want incompressible flow")
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")

        # Second interaction
        context.add_message("assistant", "Got it. What about Reynolds number?")
        context.update_nested_field("collected_data.physics.reynolds_number", 100000)

        # Third interaction
        context.add_message("user", "Around 100k")

        # Verify context preservation
        messages = context.get_conversation_history()
        assert len(messages) == 3

        flow_type = context.get_nested_field("collected_data.physics.flow_type")
        assert flow_type == "incompressible"

        reynolds = context.get_nested_field("collected_data.physics.reynolds_number")
        assert reynolds == 100000


class TestInteractiveWorkflowDataIntegration:
    """Integration tests for interactive workflow data handling."""

    def test_workflow_with_context_sync(self):
        """Test workflow data synchronization with context."""
        from openfoam_llm_wrapper.core.interactive_workflow import (
            InformationCollector,
            InteractiveWorkflowManager,
        )

        context = ConversationContext()
        context.initialize_case_info()
        mock_llm = Mock()

        collector = InformationCollector()
        manager = InteractiveWorkflowManager(context, mock_llm)

        # Update collector
        collector.update_field("physics", "flow_type", "incompressible")
        manager.collector = collector

        # Verify sync
        assert collector.get_field("physics", "flow_type") == "incompressible"

    def test_summary_generation_from_collected_data(self):
        """Test summary generation from collected case data."""
        from openfoam_llm_wrapper.core.summary_formatter import format_summary_for_display

        context = ConversationContext()
        context.initialize_case_info()

        # Collect some data
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")
        context.mark_category_complete("physics")

        # Generate summary
        summary = format_summary_for_display(context.case_info)

        # Verify summary
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_data_validation_during_collection(self):
        """Test data validation as it's being collected."""
        from openfoam_llm_wrapper.knowledge.information_schema import (
            validate_field,
        )

        # Valid data
        is_valid, error = validate_field("physics", "flow_type", "incompressible")
        assert is_valid is True
        assert error is None

        # Invalid data
        is_valid, error = validate_field("physics", "flow_type", "invalid_type")
        assert is_valid is False
        assert error is not None

    def test_missing_required_fields_detection(self):
        """Test detection of missing required fields."""
        from openfoam_llm_wrapper.knowledge.information_schema import (
            validate_category,
        )

        incomplete_data = {
            "flow_type": "incompressible",
            # Missing time_type which is required
        }

        result = validate_category("physics", incomplete_data)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_optional_fields_handling(self):
        """Test proper handling of optional fields."""
        context = ConversationContext()
        context.initialize_case_info()

        # Can skip optional fields
        context.update_nested_field("collected_data.physics.flow_type", "incompressible")
        context.update_nested_field("collected_data.physics.time_type", "steady")
        # Not updating optional fields like special_physics

        # Should still be valid if required fields are present
        physics_data = context.get_nested_field("collected_data.physics")
        assert physics_data["flow_type"] == "incompressible"
        assert physics_data["time_type"] == "steady"


class TestCaseGenerationWithValidation:
    """Integration tests for case generation with validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        if Path(temp).exists():
            shutil.rmtree(temp)

    def test_file_generation_and_structure(self, temp_dir):
        """Test generating files with proper directory structure."""
        files = {
            "system/controlDict": "application simpleFoam;",
            "system/fvSchemes": "ddtSchemes { default steadyState; }",
            "0/U": "dimensions [0 1 -1 0 0 0 0];",
            "0/p": "dimensions [0 2 -2 0 0 0 0];",
        }

        # Write files
        for filepath, content in files.items():
            full_path = Path(temp_dir) / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Verify structure
        assert (Path(temp_dir) / "system").exists()
        assert (Path(temp_dir) / "0").exists()
        assert (Path(temp_dir) / "system" / "controlDict").exists()
        assert (Path(temp_dir) / "0" / "U").exists()

    def test_generated_files_content_validation(self):
        """Test validation of generated file content."""
        files = {
            "system/controlDict": """FoamFile { version 2.0; }
application simpleFoam;
startTime 0;
endTime 1000;""",
            "0/U": """FoamFile { version 2.0; }
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);""",
        }

        # Basic validation of OpenFOAM structure
        for filename, content in files.items():
            assert "FoamFile" in content
            assert "version" in content
            assert len(content) > 20

    def test_case_readiness_check(self):
        """Test checking if case is ready for generation."""
        context = ConversationContext()
        context.initialize_case_info()

        # Mark only some categories complete
        context.mark_category_complete("physics")
        context.mark_category_complete("geometry")

        status = context.get_completion_status()

        # Check status
        assert status["physics"] is True
        assert status["geometry"] is True
        assert status["solver"] is False  # Not complete

        # Case is not fully ready
        completed_count = sum(1 for v in status.values() if v is True)
        assert completed_count < len(status)
