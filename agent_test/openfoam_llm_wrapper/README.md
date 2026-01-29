# OpenFOAM LLM Wrapper

An AI-powered assistant for CFD engineers using OpenFOAM. This tool leverages large language models (Claude, GPT-4) to help with:

- **Case Generation**: Create complete OpenFOAM cases from natural language descriptions
- **Error Explanation**: Understand cryptic OpenFOAM error messages
- **Solver Recommendations**: Get guidance on which solver to use for your simulation
- **Interactive Assistance**: Chat with an AI expert about OpenFOAM best practices

## Project Structure

```
openfoam_llm_wrapper/
├── cli/                          # Command line interface
│   └── main.py                   # CLI entry point
├── core/                         # Core business logic
│   ├── orchestrator.py          # Workflow coordinator
│   ├── intent_classifier.py     # User intent detection
│   ├── context_manager.py       # Conversation context
│   └── session.py               # Interactive session
├── llm/                          # LLM integration
│   ├── client.py                # LLM API client
│   ├── prompt_builder.py        # Prompt construction
│   └── response_parser.py       # Response parsing
├── knowledge/                    # Domain knowledge
│   ├── solvers.py               # Solver database
│   └── boundary_conditions.py   # BC types
├── generators/                   # File generation
│   └── case_generator.py        # Case creator
├── validators/                   # Validation
│   └── syntax_validator.py      # Syntax checking
├── utils/                        # Utilities
│   └── file_io.py               # File operations
└── config/                       # Configuration
    └── settings.py              # App settings
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Setup

1. **Clone or navigate to the project directory**

```bash
cd openfoam_llm_wrapper
```

2. **Install dependencies**

Using pip:
```bash
pip install -r requirements.txt
```

Or using poetry:
```bash
poetry install
```

3. **Set up API credentials**

For Claude API:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or for OpenAI:
```bash
export OPENAI_API_KEY="your-api-key-here"
export LLM_PROVIDER=openai
```

## Usage

### Interactive Chat Mode

Start an interactive conversation with the assistant:

```bash
python -m openfoam_llm_wrapper.cli.main chat -i
```

Example conversation:
```
You: Generate a steady-state incompressible flow case for a pipe with diameter 0.1m and inlet velocity 5 m/s
Assistant: I'll create a complete OpenFOAM case for incompressible pipe flow...
[Case generated successfully at: ./openfoam_case]
```

### Generate Case from Command Line

Create an OpenFOAM case from a description:

```bash
python -m openfoam_llm_wrapper.cli.main generate \
  "Steady incompressible turbulent flow in a rectangular duct with inlet velocity 2 m/s" \
  -o ./my_cases \
  -n duct_flow
```

### Explain Error Messages

Get help understanding an error:

```bash
python -m openfoam_llm_wrapper.cli.main explain "Foam::error::printStack: Could not open file ..."
```

### Recommend a Solver

Get solver recommendations for your physics:

```bash
python -m openfoam_llm_wrapper.cli.main recommend \
  "I need to simulate transient heat transfer with natural convection"
```

## Features

### Phase 1 (MVP) - Current Implementation

✅ **Foundation**
- CLI interface with multiple commands
- LLM integration (Claude/OpenAI)
- Prompt templates for domain-specific queries
- Basic logging and error handling

✅ **Core Functionality**
- Natural language case description input
- Dictionary file generation framework
- File I/O and validation
- Intent classification (generate, explain, recommend, etc)

✅ **Knowledge Base**
- Solver database with 4+ common solvers
- Boundary condition types catalog
- Domain knowledge integration

### Phase 2+ (Future Enhancements)

- Mesh quality analysis from checkMesh output
- Function object generation (probes, forces, etc)
- snappyHexMesh configuration assistance
- Parallel decomposition setup
- ParaView script generation
- RAG (Retrieval Augmented Generation) for OpenFOAM docs

## Configuration

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=anthropic              # or "openai"
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...

# Application
OUTPUT_DIR=./cases
LOG_LEVEL=INFO
STRICT_VALIDATION=true
OPENFOAM_VERSION=v2306
```

### Settings File

Edit `config/settings.py` to customize defaults:

```python
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_CASE_NAME = "openfoam_case"
LOG_LEVEL = "INFO"
```

## Development

### Project Setup for Development

```bash
# Install with dev dependencies
poetry install --with dev

# Run linting
ruff check .

# Format code
black .

# Type checking
mypy openfoam_llm_wrapper

# Run tests
pytest tests/
```

### Creating New Prompts

Add new prompt templates in `llm/prompt_builder.py`:

```python
def build_custom_prompt(self, input_text: str) -> str:
    return f"""{self.SYSTEM_PROMPT}

Your custom instruction here...

Input: {input_text}
"""
```

### Adding Knowledge Base Entries

Add solvers to `knowledge/solvers.py`:

```python
SOLVERS["myNewSolver"] = Solver(
    name="myNewSolver",
    category="incompressible",
    time_type="steady",
    description="...",
    required_fields=["U", "p"],
    ...
)
```

## Common Tasks

### Troubleshooting

**API Key Not Working**
```bash
# Check your API key is set
echo $ANTHROPIC_API_KEY

# Test LLM connection
python -c "from anthropic import Anthropic; print('OK')"
```

**File Permission Errors**
```bash
# Ensure output directory is writable
mkdir -p ./cases
chmod 755 ./cases
```

**Import Errors**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## Architecture

### Key Components

- **CLI (main.py)**: User interface with click commands
- **Orchestrator**: Coordinates workflow between LLM, generators, validators
- **LLMClient**: Abstracts API calls to Claude/OpenAI
- **PromptBuilder**: Constructs domain-specific prompts
- **CaseGenerator**: Creates OpenFOAM directory structure and files
- **SyntaxValidator**: Validates generated files
- **Knowledge Base**: Solver and BC reference database

### Data Flow

```
User Input
    ↓
CLI (Intent parsing)
    ↓
Orchestrator (Routing)
    ↓
LLMClient (API call)
    ↓
PromptBuilder (Domain context)
    ↓
LLM Response
    ↓
ResponseParser (Structure extraction)
    ↓
CaseGenerator (File creation)
    ↓
SyntaxValidator (Quality check)
    ↓
Output Files
```

## Contributing

This is an active project. To contribute:

1. Follow existing code style and patterns
2. Add tests for new features
3. Update documentation
4. Use type hints

## License

MIT

## Support

For issues and feature requests, please refer to the project's issue tracker.

## Roadmap

- **Week 1-2**: Foundation and basic CLI
- **Week 3-4**: Full case generation with multiple solver templates
- **Week 5**: Error explanation and debugging
- **Week 6**: Polish, testing, documentation

## References

- [OpenFOAM Documentation](https://www.openfoam.com/)
- [OpenFOAM Tutorials](https://www.openfoam.com/documentation/tutorials)
- [CFD Tutorials](https://openfoamwiki.net/index.php/Main_Page)