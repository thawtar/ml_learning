"""File I/O utilities."""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Dictionary with YAML contents
    """
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {str(e)}")
        return {}


def write_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Write data to a YAML file.

    Args:
        data: Dictionary to write
        file_path: Path to write to
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.debug(f"Wrote YAML file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing YAML file {file_path}: {str(e)}")


def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with JSON contents
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {str(e)}")
        return {}


def write_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Write data to a JSON file.

    Args:
        data: Dictionary to write
        file_path: Path to write to
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Wrote JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {str(e)}")


def read_file(file_path: str) -> str:
    """
    Read a text file.

    Args:
        file_path: Path to the file

    Returns:
        File contents as string
    """
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return ""


def write_file(content: str, file_path: str) -> None:
    """
    Write content to a file.

    Args:
        content: Content to write
        file_path: Path to write to
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        logger.debug(f"Wrote file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {str(e)}")
