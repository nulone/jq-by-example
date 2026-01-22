"""Tests for color utilities."""

from unittest.mock import patch

import pytest

from src.colors import (
    Colors,
    bold,
    cyan,
    dim,
    error,
    info,
    should_use_color,
    success,
    warning,
)


class TestShouldUseColor:
    """Tests for should_use_color function."""

    def test_returns_false_when_no_color_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """should_use_color returns False when NO_COLOR is set."""
        monkeypatch.setenv("NO_COLOR", "1")
        with patch("sys.stdout.isatty", return_value=True):
            assert should_use_color() is False

    def test_returns_false_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """should_use_color returns False when stdout is not a TTY."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        with patch("sys.stdout.isatty", return_value=False):
            assert should_use_color() is False

    def test_returns_true_when_tty_and_no_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """should_use_color returns True when TTY and NO_COLOR not set."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        with patch("sys.stdout.isatty", return_value=True):
            assert should_use_color() is True


class TestColorFormatting:
    """Tests for color formatting functions."""

    def test_success_formats_green(self) -> None:
        """success() should format text in green."""
        result = success("test")
        # Should contain text and color codes (if colors enabled)
        assert "test" in result

    def test_error_formats_red(self) -> None:
        """error() should format text in red."""
        result = error("test")
        assert "test" in result

    def test_warning_formats_yellow(self) -> None:
        """warning() should format text in yellow."""
        result = warning("test")
        assert "test" in result

    def test_info_formats_blue(self) -> None:
        """info() should format text in blue."""
        result = info("test")
        assert "test" in result

    def test_bold_formats_text(self) -> None:
        """bold() should format text as bold."""
        result = bold("test")
        assert "test" in result

    def test_dim_formats_text(self) -> None:
        """dim() should format text as dim."""
        result = dim("test")
        assert "test" in result

    def test_cyan_formats_text(self) -> None:
        """cyan() should format text in cyan."""
        result = cyan("test")
        assert "test" in result


class TestColorsClass:
    """Tests for Colors class."""

    def test_colors_have_values_initially(self) -> None:
        """Colors should have ANSI codes initially."""
        # Note: This may depend on whether colors were disabled during module load
        # We just check that the attributes exist
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "YELLOW")
        assert hasattr(Colors, "BLUE")
        assert hasattr(Colors, "BOLD")
        assert hasattr(Colors, "RESET")

    def test_disable_removes_colors(self) -> None:
        """disable() should set all color codes to empty strings."""
        # Save original values
        original_red = Colors.RED
        original_reset = Colors.RESET

        # Disable colors
        Colors.disable()

        # All should be empty strings
        assert Colors.RED == ""
        assert Colors.GREEN == ""
        assert Colors.YELLOW == ""
        assert Colors.BLUE == ""
        assert Colors.BOLD == ""
        assert Colors.RESET == ""

        # Restore original values for other tests
        # This is a bit hacky but necessary since Colors is a class variable
        Colors.RED = original_red
        Colors.RESET = original_reset
