"""Generate complete OpenFOAM case directories."""

import os
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class CaseGenerator:
    """Orchestrates creation of complete OpenFOAM case structures."""

    def create_case(
        self, files: Dict[str, str], output_dir: str = ".", case_name: str = "openfoam_case"
    ) -> str:
        """
        Create a complete OpenFOAM case directory structure.

        Args:
            files: Dictionary mapping file paths to contents
            output_dir: Directory to create the case in
            case_name: Name of the case directory

        Returns:
            Full path to created case directory
        """
        # Create case root directory
        case_path = Path(output_dir) / case_name
        case_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        required_dirs = ["0", "constant", "system"]
        for dir_name in required_dirs:
            (case_path / dir_name).mkdir(exist_ok=True)

        # Write files
        files_written = 0
        for file_path, content in files.items():
            full_path = case_path / file_path

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(full_path, "w") as f:
                f.write(content)

            files_written += 1
            logger.debug(f"Wrote file: {file_path}")

        logger.info(f"Created case '{case_name}' at {case_path} with {files_written} files")

        return str(case_path)

    def create_empty_case(
        self, output_dir: str = ".", case_name: str = "openfoam_case"
    ) -> str:
        """
        Create an empty OpenFOAM case structure with standard directories.

        Args:
            output_dir: Directory to create the case in
            case_name: Name of the case directory

        Returns:
            Full path to created case directory
        """
        case_path = Path(output_dir) / case_name
        case_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        dirs = {
            "0": "Initial conditions",
            "constant": "Physical properties and mesh data",
            "system": "Solver settings and schemes",
        }

        for dir_name, description in dirs.items():
            dir_path = case_path / dir_name
            dir_path.mkdir(exist_ok=True)

            # Create placeholder file with description
            placeholder = dir_path / ".placeholder"
            with open(placeholder, "w") as f:
                f.write(f"{description}\n")

        logger.info(f"Created empty case structure at {case_path}")

        return str(case_path)
