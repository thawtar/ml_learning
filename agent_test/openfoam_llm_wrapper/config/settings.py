"""Configuration settings for OpenFOAM LLM Wrapper."""

import os
from pathlib import Path


class Settings:
    """Application settings."""

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model selection
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    OPENAI_MODEL: str = "gpt-4"

    # Output configuration
    DEFAULT_OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", ".")
    DEFAULT_CASE_NAME: str = "openfoam_case"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str | None = os.getenv("LOG_FILE", None)

    # Validation settings
    STRICT_VALIDATION: bool = os.getenv("STRICT_VALIDATION", "true").lower() == "true"

    # OpenFOAM version
    OPENFOAM_VERSION: str = os.getenv("OPENFOAM_VERSION", "v2306")

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"

    @classmethod
    def get_llm_key(cls) -> str:
        """Get the appropriate LLM API key."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_API_KEY
        return cls.ANTHROPIC_API_KEY

    @classmethod
    def get_model_name(cls) -> str:
        """Get the appropriate model name."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_MODEL
        return cls.CLAUDE_MODEL
