"""Parse LLM responses into structured data."""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse LLM responses into structured file format."""

    def parse_case_files(self, response_text: str) -> Dict[str, str]:
        """
        Parse LLM response into OpenFOAM case files.

        Expects response to contain file blocks in format:
        <file path="path/to/file">
        file contents
        </file>

        Args:
            response_text: The LLM response text

        Returns:
            Dictionary mapping file paths to file contents
        """
        files = {}

        # Pattern to match <file> tags
        pattern = r'<file\s+path="([^"]+)">([^<]*?)</file>'

        matches = re.findall(pattern, response_text, re.DOTALL)

        for file_path, content in matches:
            # Clean up content
            content = content.strip()

            # Normalize path separators
            file_path = file_path.strip()

            files[file_path] = content
            logger.debug(f"Parsed file: {file_path} ({len(content)} chars)")

        logger.info(f"Parsed {len(files)} files from LLM response")

        return files

    def extract_text_blocks(self, response_text: str) -> Dict[str, str]:
        """
        Extract code blocks marked with triple backticks.

        Args:
            response_text: The response text

        Returns:
            Dictionary mapping block titles to contents
        """
        blocks = {}

        # Pattern for markdown code blocks
        pattern = r"```(\w+)?\n(.*?)\n```"

        matches = re.findall(pattern, response_text, re.DOTALL)

        for i, (language, content) in enumerate(matches):
            key = f"block_{i}" if not language else language
            blocks[key] = content.strip()

        return blocks
