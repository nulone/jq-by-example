"""
ANSI color codes for terminal output.

This module provides color formatting utilities for CLI output, with automatic
detection of terminal capabilities and support for the NO_COLOR standard.
"""

import os
import sys


class Colors:
    """ANSI color codes for terminal formatting."""

    # Basic colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY or --no-color)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RESET = ""


def should_use_color() -> bool:
    """
    Check if colors should be used based on environment.

    Returns:
        True if colors should be enabled, False otherwise.
    """
    # Check NO_COLOR env var (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False

    # Check if stdout is a TTY
    return sys.stdout.isatty()


# Initialize colors based on environment
if not should_use_color():
    Colors.disable()


def success(text: str) -> str:
    """Format text as success (green)."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def error(text: str) -> str:
    """Format text as error (red)."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def warning(text: str) -> str:
    """Format text as warning (yellow)."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def info(text: str) -> str:
    """Format text as info (blue)."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def bold(text: str) -> str:
    """Format text as bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def dim(text: str) -> str:
    """Format text as dim."""
    return f"{Colors.DIM}{text}{Colors.RESET}"


def cyan(text: str) -> str:
    """Format text in cyan."""
    return f"{Colors.CYAN}{text}{Colors.RESET}"
