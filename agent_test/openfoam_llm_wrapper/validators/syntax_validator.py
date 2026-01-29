"""Validate OpenFOAM file syntax and consistency."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validation check."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid: bool = True

    def add_error(self, message: str) -> None:
        """Add an error to the result."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SyntaxValidator:
    """Validates OpenFOAM case file syntax."""

    REQUIRED_FILES = [
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "constant/transportProperties",
        "0/U",
        "0/p",
    ]

    def validate_case(self, case_path: str) -> Dict[str, Any]:
        """
        Validate an OpenFOAM case directory.

        Args:
            case_path: Path to the case directory

        Returns:
            Validation result dictionary
        """
        result = ValidationResult()
        case_path = Path(case_path)

        logger.info(f"Validating case at {case_path}")

        # Check if case path exists
        if not case_path.exists():
            result.add_error(f"Case path does not exist: {case_path}")
            return result.to_dict()

        # Check required directories
        for dir_name in ["0", "constant", "system"]:
            if not (case_path / dir_name).is_dir():
                result.add_error(f"Required directory missing: {dir_name}/")

        # Check required files (optional for MVP)
        existing_files = []
        for file_path in case_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(case_path)
                existing_files.append(str(relative_path))

        if not existing_files:
            result.add_warning("Case directory is empty - no files found")

        # Basic syntax validation
        for file_path in existing_files:
            full_path = case_path / file_path
            self._validate_file_syntax(full_path, result)

        logger.info(f"Validation complete: is_valid={result.is_valid}")

        return result.to_dict()

    def _validate_file_syntax(self, file_path: Path, result: ValidationResult) -> None:
        """
        Validate syntax of a single OpenFOAM file.

        Args:
            file_path: Path to the file
            result: ValidationResult object to record issues
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic checks
            if not content.strip():
                result.add_warning(f"Empty file: {file_path.name}")
                return

            # Check for balanced brackets (basic check)
            open_brackets = content.count("{")
            close_brackets = content.count("}")

            if open_brackets != close_brackets:
                result.add_error(
                    f"Unbalanced brackets in {file_path.name}: "
                    f"{open_brackets} {{ vs {close_brackets} }}"
                )

            # Check for required semicolons in dict files
            if file_path.suffix == "" or file_path.name in [
                "controlDict",
                "fvSchemes",
                "fvSolution",
                "transportProperties",
            ]:
                if "{" in content and ";" not in content:
                    result.add_warning(
                        f"No semicolons found in {file_path.name} - may be incomplete"
                    )

        except Exception as e:
            result.add_error(f"Error validating {file_path.name}: {str(e)}")

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single file.

        Args:
            file_path: Path to the file

        Returns:
            Validation result dictionary
        """
        result = ValidationResult()
        full_path = Path(file_path)

        if not full_path.exists():
            result.add_error(f"File not found: {file_path}")
        else:
            self._validate_file_syntax(full_path, result)

        return result.to_dict()
