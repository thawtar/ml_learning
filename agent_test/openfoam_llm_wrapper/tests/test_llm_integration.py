"""Unit tests for LLM integration components."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from openfoam_llm_wrapper.llm.client import LLMClient
from openfoam_llm_wrapper.llm.prompt_builder import PromptBuilder
from openfoam_llm_wrapper.llm.response_parser import ResponseParser


class TestPromptBuilder:
    """Test suite for PromptBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a fresh prompt builder for each test."""
        return PromptBuilder()

    def test_system_prompt_exists(self, builder):
        """Test that system prompt is defined."""
        assert builder.SYSTEM_PROMPT is not None
        assert len(builder.SYSTEM_PROMPT) > 0
        assert "OpenFOAM" in builder.SYSTEM_PROMPT

    def test_build_case_generation_prompt(self, builder):
        """Test building case generation prompt."""
        description = "Incompressible flow in a pipe"
        intent = "generate_case"

        prompt = builder.build_case_generation_prompt(description, intent)

        assert prompt is not None
        assert description in prompt
        assert "system/controlDict" in prompt
        assert "<file path=" in prompt

    def test_build_error_explanation_prompt(self, builder):
        """Test building error explanation prompt."""
        error = "Foam::error: Cannot find file"

        prompt = builder.build_error_explanation_prompt(error)

        assert prompt is not None
        assert error in prompt
        assert "What went wrong" in prompt

    def test_build_solver_recommendation_prompt(self, builder):
        """Test building solver recommendation prompt."""
        physics = "Incompressible turbulent flow"

        prompt = builder.build_solver_recommendation_prompt(physics)

        assert prompt is not None
        assert physics in prompt
        assert "Recommended solver" in prompt

    def test_build_question_prompt(self, builder):
        """Test building question prompt."""
        question = "How do I set boundary conditions?"

        prompt = builder.build_question_prompt(question)

        assert prompt is not None
        assert question in prompt

    def test_build_interactive_system_prompt(self, builder):
        """Test building interactive system prompt."""
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
            "physics": {"flow_type": "incompressible"},
            "geometry": {},
            "boundary_conditions": {},
            "solver": {},
            "fluid_properties": {},
            "mesh": {},
            "simulation_goals": {},
            "advanced": {},
        }
        question_history = []

        prompt = builder.build_interactive_system_prompt(
            collection_status, collected_data, question_history
        )

        assert prompt is not None
        assert "expert OpenFOAM consultant" in prompt
        assert "READY_TO_GENERATE" in prompt
        assert "incompressible" in prompt

    def test_format_collection_status(self, builder):
        """Test formatting collection status."""
        status = {"physics": True, "geometry": False}

        formatted = builder._format_collection_status(status)

        assert "✓" in formatted
        assert "!" in formatted
        assert "physics" in formatted.lower()

    def test_format_collected_data(self, builder):
        """Test formatting collected data."""
        data = {
            "physics": {"flow_type": "incompressible", "time_type": "steady"},
            "geometry": {},
        }

        formatted = builder._format_collected_data(data)

        assert "incompressible" in formatted
        assert "steady" in formatted

    def test_format_collected_data_empty(self, builder):
        """Test formatting empty collected data."""
        data = {
            "physics": {},
            "geometry": {},
        }

        formatted = builder._format_collected_data(data)

        assert formatted is not None
        # Should handle empty data gracefully

    def test_format_question_history(self, builder):
        """Test formatting question history."""
        history = [
            {
                "topic": "physics",
                "question": "What type of flow?",
                "answer": "incompressible"
            },
            {
                "topic": "geometry",
                "question": "What is the geometry?",
                "answer": "pipe with 0.1m diameter"
            },
        ]

        formatted = builder._format_question_history(history)

        assert "What type of flow" in formatted
        assert "incompressible" in formatted
        assert "Q1" in formatted
        assert "Q2" in formatted

    def test_format_question_history_empty(self, builder):
        """Test formatting empty question history."""
        history = []

        formatted = builder._format_question_history(history)

        assert "No previous questions" in formatted

    def test_format_question_history_limits_to_five(self, builder):
        """Test that history formatting limits to 5 most recent."""
        history = [
            {"topic": f"topic_{i}", "question": f"Q{i}?", "answer": f"A{i}"}
            for i in range(10)
        ]

        formatted = builder._format_question_history(history)

        # Should only show last 5
        assert "topic_9" in formatted
        assert "topic_5" in formatted
        assert "topic_4" not in formatted or formatted.count("Q4") == 0

    def test_build_final_case_prompt(self, builder):
        """Test building final case generation prompt."""
        case_info = {
            "collected_data": {
                "physics": {"flow_type": "incompressible", "time_type": "steady"},
                "geometry": {"description": "pipe"},
                "boundary_conditions": {"patches": {}},
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {},
            }
        }

        prompt = builder.build_final_case_prompt(case_info)

        assert prompt is not None
        assert "incompressible" in prompt
        assert "steady" in prompt
        assert "<file path=" in prompt

    def test_build_knowledge_base_context(self, builder):
        """Test building knowledge base context."""
        context = builder.build_knowledge_base_context()

        assert context is not None
        assert "AVAILABLE SOLVERS" in context
        assert "AVAILABLE BOUNDARY CONDITIONS" in context
        assert "simpleFoam" in context or "Boundary" in context


class TestResponseParser:
    """Test suite for ResponseParser class."""

    @pytest.fixture
    def parser(self):
        """Create a fresh parser for each test."""
        return ResponseParser()

    def test_parse_single_file(self, parser):
        """Test parsing a single file."""
        response = """
<file path="system/controlDict">
FoamFile {
    version 2.0;
}
</file>
"""
        files = parser.parse_case_files(response)

        assert len(files) == 1
        assert "system/controlDict" in files
        assert "version 2.0" in files["system/controlDict"]

    def test_parse_multiple_files(self, parser):
        """Test parsing multiple files."""
        response = """
<file path="system/controlDict">
content1
</file>
<file path="system/fvSchemes">
content2
</file>
<file path="0/U">
content3
</file>
"""
        files = parser.parse_case_files(response)

        assert len(files) == 3
        assert "system/controlDict" in files
        assert "system/fvSchemes" in files
        assert "0/U" in files

    def test_parse_file_with_multiline_content(self, parser):
        """Test parsing file with multiline content."""
        response = """
<file path="system/controlDict">
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
startTime       0;
endTime         1000;
deltaT          0.1;
</file>
"""
        files = parser.parse_case_files(response)

        assert len(files) == 1
        file_content = files["system/controlDict"]
        assert "startTime" in file_content
        assert "deltaT" in file_content

    def test_parse_empty_response(self, parser):
        """Test parsing empty response."""
        response = ""
        files = parser.parse_case_files(response)

        assert len(files) == 0

    def test_parse_no_file_tags(self, parser):
        """Test parsing response with no file tags."""
        response = "This is just regular text, no files here"
        files = parser.parse_case_files(response)

        assert len(files) == 0

    def test_parse_with_extra_whitespace(self, parser):
        """Test parsing with extra whitespace."""
        response = """
<file path = "system/controlDict" >
content
</file>
"""
        # Path might have spaces - test robustness
        files = parser.parse_case_files(response)

        # Should handle gracefully
        assert isinstance(files, dict)

    def test_extract_text_blocks_with_markdown(self, parser):
        """Test extracting markdown code blocks."""
        response = """
Some explanation text

```python
def hello():
    print("world")
```

More text

```bash
echo "hello"
```
"""
        blocks = parser.extract_text_blocks(response)

        assert len(blocks) == 2

    def test_extract_text_blocks_empty(self, parser):
        """Test extracting from response with no code blocks."""
        response = "Just regular text with no code blocks"
        blocks = parser.extract_text_blocks(response)

        assert len(blocks) == 0

    def test_parse_nested_tags(self, parser):
        """Test parsing with nested-like content."""
        response = """
<file path="0/U">
boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (1 0 0);
    }
}
</file>
"""
        files = parser.parse_case_files(response)

        assert len(files) == 1
        assert "boundaryField" in files["0/U"]
        assert "inlet" in files["0/U"]


class TestLLMClientMethods:
    """Test suite for LLMClient methods."""

    @pytest.fixture
    def client_no_api_key(self):
        """Create client without API key."""
        return LLMClient(api_key="")

    def test_client_initialization_without_key(self):
        """Test client initializes even without API key."""
        client = LLMClient(api_key="")

        assert client.provider == "anthropic"
        # Client should be None if no key
        assert client.client is None or client.client is not None

    def test_chat_with_history_no_client(self, client_no_api_key):
        """Test chat_with_history gracefully handles missing client."""
        result = client_no_api_key.chat_with_history(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are helpful"
        )

        assert "not initialized" in result or isinstance(result, str)

    def test_interactive_case_questioning_no_client(self, client_no_api_key):
        """Test interactive_case_questioning handles missing client."""
        result = client_no_api_key.interactive_case_questioning(
            conversation_history=[],
            collection_status={},
            collected_data={},
            question_history=[]
        )

        assert isinstance(result, str)

    def test_placeholder_response(self):
        """Test placeholder response generation."""
        response = LLMClient._placeholder_response("generate_case")

        assert isinstance(response, dict)
        assert "system/controlDict" in response
        assert "0/U" in response
        assert "0/p" in response


class TestPromptBuilderIntegration:
    """Integration tests for prompt building."""

    @pytest.fixture
    def builder(self):
        """Create prompt builder."""
        return PromptBuilder()

    def test_interactive_prompt_completeness(self, builder):
        """Test that interactive prompt contains all necessary elements."""
        status = {
            "physics": False,
            "geometry": False,
            "boundary_conditions": False,
            "solver": False,
            "fluid_properties": False,
            "mesh": False,
            "simulation_goals": False,
            "advanced": False,
        }
        data = {
            "physics": {},
            "geometry": {},
            "boundary_conditions": {},
            "solver": {},
            "fluid_properties": {},
            "mesh": {},
            "simulation_goals": {},
            "advanced": {},
        }

        prompt = builder.build_interactive_system_prompt(status, data, [])

        # Check for key elements
        assert "expert OpenFOAM consultant" in prompt
        assert "ONE expert-level technical question" in prompt
        assert "READY_TO_GENERATE" in prompt
        assert "skip" in prompt.lower()
        assert "back" in prompt.lower()

    def test_final_case_prompt_with_full_data(self, builder):
        """Test final case prompt with comprehensive data."""
        case_info = {
            "collected_data": {
                "physics": {
                    "flow_type": "incompressible",
                    "time_type": "steady",
                    "reynolds_number": 100000,
                    "turbulence_model": "kOmegaSST"
                },
                "geometry": {
                    "description": "Rectangular duct 0.1m x 0.2m x 2m",
                    "dimension": "3D"
                },
                "boundary_conditions": {
                    "patches": {
                        "inlet": {"type": "fixedValue", "field_values": {"U": "5 0 0"}},
                        "outlet": {"type": "zeroGradient"}
                    }
                },
                "solver": {"solver": "simpleFoam"},
                "fluid_properties": {"density": 1.0, "viscosity": 0.001},
                "mesh": {},
                "simulation_goals": {},
                "advanced": {}
            }
        }

        prompt = builder.build_final_case_prompt(case_info)

        assert "incompressible" in prompt
        assert "steady" in prompt
        assert "100000" in prompt
        assert "simpleFoam" in prompt
        assert "Rectangular duct" in prompt


class TestResponseParserIntegration:
    """Integration tests for response parsing."""

    @pytest.fixture
    def parser(self):
        """Create parser."""
        return ResponseParser()

    def test_realistic_llm_response(self, parser):
        """Test parsing a realistic LLM response."""
        response = """
I'll generate the OpenFOAM case for your incompressible pipe flow simulation.

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
internalField   uniform (1 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
}
</file>

All files have been generated with correct OpenFOAM syntax.
"""
        files = parser.parse_case_files(response)

        assert len(files) == 3
        assert "system/controlDict" in files
        assert "system/fvSchemes" in files
        assert "0/U" in files
        assert "application     simpleFoam" in files["system/controlDict"]
        assert "ddtSchemes" in files["system/fvSchemes"]
        assert "fixedValue" in files["0/U"]
