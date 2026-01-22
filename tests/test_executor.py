"""
Integration tests for JQ binary execution including error handling.

This module tests the JQExecutor class for proper execution of jq filters,
handling of various edge cases, and proper error reporting.
"""

import pytest

from src.executor import JQExecutor


class TestJQExecutorInit:
    """Tests for JQExecutor initialization and jq binary detection."""

    def test_missing_jq_raises_runtime_error(self):
        """Constructor fails if jq binary not found at specified path."""
        with pytest.raises(RuntimeError) as exc_info:
            JQExecutor(jq_path="/nonexistent/path/to/jq")

        assert "jq binary not found" in str(exc_info.value)
        assert "/nonexistent/path/to/jq" in str(exc_info.value)

    def test_default_jq_path_resolved(self, executor: JQExecutor):
        """Default jq path is resolved to absolute path."""
        # If we got here, jq was found (fixture skips otherwise)
        assert executor.jq_path is not None
        assert len(executor.jq_path) > 0

    def test_custom_timeout_accepted(self):
        """Custom timeout value is accepted and stored."""
        try:
            exec = JQExecutor(timeout_sec=5.0)
            assert exec.timeout_sec == 5.0
        except RuntimeError:
            pytest.skip("jq binary not available")

    def test_custom_max_output_bytes_accepted(self):
        """Custom max output bytes value is accepted and stored."""
        try:
            exec = JQExecutor(max_output_bytes=500_000)
            assert exec.max_output_bytes == 500_000
        except RuntimeError:
            pytest.skip("jq binary not available")


class TestSimpleFilters:
    """Tests for basic jq filter execution."""

    def test_simple_field_extraction(self, executor: JQExecutor):
        """Basic .field extraction works correctly."""
        result = executor.run(".foo", {"foo": "bar"})

        assert result.is_success is True
        assert result.exit_code == 0
        assert result.is_timeout is False
        assert result.stdout == '"bar"'
        assert result.stderr == ""

    def test_numeric_field_extraction(self, executor: JQExecutor):
        """Numeric field extraction returns unquoted number."""
        result = executor.run(".count", {"count": 42})

        assert result.is_success is True
        assert result.stdout == "42"

    def test_boolean_field_extraction(self, executor: JQExecutor):
        """Boolean field extraction returns true/false."""
        result = executor.run(".active", {"active": True})

        assert result.is_success is True
        assert result.stdout == "true"

    def test_identity_filter(self, executor: JQExecutor):
        """Identity filter . returns the input unchanged."""
        result = executor.run(".", {"x": 1, "y": 2})

        assert result.is_success is True
        # jq -c produces compact output
        assert "x" in result.stdout
        assert "y" in result.stdout


class TestArrayFilters:
    """Tests for array-related jq filters."""

    def test_array_iteration(self, executor: JQExecutor):
        """Array iteration with .[].field produces multiple lines."""
        result = executor.run(".[].x", [{"x": 1}, {"x": 2}, {"x": 3}])

        assert result.is_success is True
        # Each value on separate line, newlines stripped at end
        lines = result.stdout.split("\n")
        assert lines == ["1", "2", "3"]

    def test_array_index_access(self, executor: JQExecutor):
        """Array index access .[n] works correctly."""
        result = executor.run(".[1]", ["a", "b", "c"])

        assert result.is_success is True
        assert result.stdout == '"b"'

    def test_array_slice(self, executor: JQExecutor):
        """Array slicing .[start:end] works correctly."""
        result = executor.run(".[1:3]", [1, 2, 3, 4, 5])

        assert result.is_success is True
        assert result.stdout == "[2,3]"

    def test_array_length(self, executor: JQExecutor):
        """Array length filter works correctly."""
        result = executor.run("length", [1, 2, 3, 4])

        assert result.is_success is True
        assert result.stdout == "4"

    def test_empty_array(self, executor: JQExecutor):
        """Empty array is handled correctly."""
        result = executor.run(".", [])

        assert result.is_success is True
        assert result.stdout == "[]"

    def test_large_array_output(self, executor: JQExecutor):
        """Large array output is handled within limits."""
        # Create array that produces reasonable output
        large_input = list(range(100))
        result = executor.run(".", large_input)

        assert result.is_success is True
        assert "0" in result.stdout
        assert "99" in result.stdout


class TestSyntaxErrors:
    """Tests for jq syntax error handling."""

    def test_invalid_brackets_syntax_error(self, executor: JQExecutor):
        """Invalid bracket syntax returns error result."""
        result = executor.run("invalid[[[", {})

        assert result.is_success is False
        assert result.exit_code != 0
        assert result.is_timeout is False
        assert len(result.stderr) > 0

    def test_unclosed_string_syntax_error(self, executor: JQExecutor):
        """Unclosed string returns error result."""
        result = executor.run('."unclosed', {})

        assert result.is_success is False
        assert result.exit_code != 0

    def test_invalid_pipe_syntax_error(self, executor: JQExecutor):
        """Invalid pipe usage returns error result."""
        result = executor.run("| |", {})

        assert result.is_success is False
        assert result.exit_code != 0

    def test_unknown_function_error(self, executor: JQExecutor):
        """Unknown function call returns error result."""
        result = executor.run("nonexistent_function", {})

        assert result.is_success is False
        assert result.exit_code != 0


class TestNestedAccess:
    """Tests for nested object access."""

    def test_nested_field_access(self, executor: JQExecutor):
        """Nested field access .a.b.c works correctly."""
        result = executor.run(".a.b.c", {"a": {"b": {"c": "deep"}}})

        assert result.is_success is True
        assert result.stdout == '"deep"'

    def test_nested_access_missing_key_returns_null(self, executor: JQExecutor):
        """Accessing missing nested key returns null."""
        result = executor.run(".a.b.c", {"a": {}})

        assert result.is_success is True
        assert result.stdout == "null"

    def test_nested_access_partial_path(self, executor: JQExecutor):
        """Partial path access returns intermediate object."""
        result = executor.run(".a.b", {"a": {"b": {"c": 1, "d": 2}}})

        assert result.is_success is True
        assert "c" in result.stdout
        assert "d" in result.stdout

    def test_deeply_nested_access(self, executor: JQExecutor):
        """Very deep nesting is handled correctly."""
        deep_obj = {"l1": {"l2": {"l3": {"l4": {"l5": "value"}}}}}
        result = executor.run(".l1.l2.l3.l4.l5", deep_obj)

        assert result.is_success is True
        assert result.stdout == '"value"'


class TestSelectFilter:
    """Tests for select filter operations."""

    def test_select_filter_basic(self, executor: JQExecutor):
        """Basic select filter works correctly."""
        result = executor.run(
            ".[] | select(.active == true)",
            [{"name": "a", "active": True}, {"name": "b", "active": False}],
        )

        assert result.is_success is True
        assert "a" in result.stdout
        assert "b" not in result.stdout

    def test_select_filter_numeric_comparison(self, executor: JQExecutor):
        """Select with numeric comparison works."""
        result = executor.run(
            ".[] | select(.value > 5)",
            [{"value": 3}, {"value": 7}, {"value": 10}],
        )

        assert result.is_success is True
        lines = result.stdout.split("\n")
        assert len(lines) == 2  # Two objects match

    def test_select_filter_no_matches(self, executor: JQExecutor):
        """Select with no matches returns empty output."""
        result = executor.run(
            ".[] | select(.x == 999)",
            [{"x": 1}, {"x": 2}],
        )

        assert result.is_success is True
        assert result.stdout == ""

    def test_select_with_string_match(self, executor: JQExecutor):
        """Select with string matching works."""
        result = executor.run(
            '.[] | select(.name == "Bob")',
            [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}],
        )

        assert result.is_success is True
        assert "Bob" in result.stdout
        assert "Alice" not in result.stdout


class TestNullHandling:
    """Tests for null value handling."""

    def test_null_input(self, executor: JQExecutor):
        """Running filter on null input works."""
        result = executor.run(".", None)

        assert result.is_success is True
        assert result.stdout == "null"

    def test_field_access_on_null(self, executor: JQExecutor):
        """Accessing field on null returns null."""
        result = executor.run(".foo", None)

        assert result.is_success is True
        assert result.stdout == "null"

    def test_null_field_value(self, executor: JQExecutor):
        """Extracting null field value works."""
        result = executor.run(".value", {"value": None})

        assert result.is_success is True
        assert result.stdout == "null"

    def test_null_in_array(self, executor: JQExecutor):
        """Null values in arrays are preserved."""
        result = executor.run(".", [1, None, 3])

        assert result.is_success is True
        assert "null" in result.stdout


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_object_identity(self, executor: JQExecutor):
        """Identity filter on empty object returns empty object."""
        result = executor.run(".", {})

        assert result.is_success is True
        assert result.stdout == "{}"

    def test_empty_object_keys(self, executor: JQExecutor):
        """Keys of empty object returns empty array."""
        result = executor.run("keys", {})

        assert result.is_success is True
        assert result.stdout == "[]"

    def test_field_access_on_empty_object(self, executor: JQExecutor):
        """Field access on empty object returns null."""
        result = executor.run(".foo", {})

        assert result.is_success is True
        assert result.stdout == "null"


class TestInputSerialization:
    """Tests for input data serialization edge cases."""

    def test_string_input(self, executor: JQExecutor):
        """String input is handled correctly."""
        result = executor.run(".", "hello")

        assert result.is_success is True
        assert result.stdout == '"hello"'

    def test_numeric_input(self, executor: JQExecutor):
        """Numeric input is handled correctly."""
        result = executor.run(". + 1", 41)

        assert result.is_success is True
        assert result.stdout == "42"

    def test_boolean_input(self, executor: JQExecutor):
        """Boolean input is handled correctly."""
        result = executor.run("not", True)

        assert result.is_success is True
        assert result.stdout == "false"

    def test_unicode_input(self, executor: JQExecutor):
        """Unicode characters in input are handled correctly."""
        result = executor.run(".name", {"name": "日本語"})

        assert result.is_success is True
        assert "日本語" in result.stdout

    def test_special_characters_in_string(self, executor: JQExecutor):
        """Special characters in strings are handled correctly."""
        result = executor.run(".text", {"text": 'Hello\n"World"\ttab'})

        assert result.is_success is True
        # The output should contain escaped versions
        assert result.exit_code == 0


class TestTimeoutHandling:
    """Tests for execution timeout handling."""

    def test_timeout_result_properties(self):
        """Timeout result has correct properties."""
        try:
            # Very short timeout to potentially trigger
            executor = JQExecutor(timeout_sec=0.001)
        except RuntimeError:
            pytest.skip("jq binary not available")

        # Note: This test might be flaky depending on system speed
        # A simple filter should complete even with very short timeout
        # We're mainly testing the timeout mechanism exists
        result = executor.run(".", {"x": 1})

        # Either it completes or times out - both are valid
        if result.is_timeout:
            assert result.exit_code == 124
            assert "timed out" in result.stderr.lower()
        else:
            assert result.is_success is True


class TestOutputSizeLimits:
    """Tests for output size limit handling."""

    def test_output_within_limits(self, executor: JQExecutor):
        """Normal output within limits succeeds."""
        # Generate reasonable sized output
        data = {"items": list(range(100))}
        result = executor.run(".", data)

        assert result.is_success is True
        assert result.exit_code == 0

    def test_small_output_limit(self):
        """Output exceeding small limit is truncated."""
        try:
            executor = JQExecutor(max_output_bytes=50)
        except RuntimeError:
            pytest.skip("jq binary not available")

        # Generate output larger than 50 bytes
        data = {"a": "x" * 100}
        result = executor.run(".", data)

        # Should be truncated
        assert result.exit_code == 137
        assert "Output too large" in result.stderr


class TestComplexFilters:
    """Tests for more complex jq filter expressions."""

    def test_map_filter(self, executor: JQExecutor):
        """Map filter transforms array elements."""
        result = executor.run("map(. * 2)", [1, 2, 3])

        assert result.is_success is True
        assert result.stdout == "[2,4,6]"

    def test_pipe_chain(self, executor: JQExecutor):
        """Piped filter chain works correctly."""
        result = executor.run(".items | .[0] | .name", {"items": [{"name": "first"}]})

        assert result.is_success is True
        assert result.stdout == '"first"'

    def test_object_construction(self, executor: JQExecutor):
        """Object construction syntax works."""
        result = executor.run("{x: .a, y: .b}", {"a": 1, "b": 2})

        assert result.is_success is True
        assert '"x":1' in result.stdout or '"x": 1' in result.stdout

    def test_array_construction(self, executor: JQExecutor):
        """Array construction syntax works."""
        result = executor.run("[.a, .b, .c]", {"a": 1, "b": 2, "c": 3})

        assert result.is_success is True
        assert result.stdout == "[1,2,3]"

    def test_conditional_expression(self, executor: JQExecutor):
        """If-then-else expression works."""
        result = executor.run('if . > 5 then "big" else "small" end', 10)

        assert result.is_success is True
        assert result.stdout == '"big"'

    def test_addition_operator(self, executor: JQExecutor):
        """Addition operator works on numbers."""
        result = executor.run(".a + .b", {"a": 10, "b": 20})

        assert result.is_success is True
        assert result.stdout == "30"

    def test_string_concatenation(self, executor: JQExecutor):
        """String concatenation works."""
        result = executor.run('.first + " " + .last', {"first": "John", "last": "Doe"})

        assert result.is_success is True
        assert result.stdout == '"John Doe"'
