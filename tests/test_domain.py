"""
Unit tests for domain data structures and their properties.

This module tests the core domain types including ExecutionResult, Attempt,
and ErrorType enum to ensure their properties and behaviors work correctly.
"""

import pytest

from src.domain import Attempt, ErrorType, Example, ExampleResult, ExecutionResult, Solution, Task


class TestExecutionResult:
    """Tests for ExecutionResult dataclass and is_success property."""

    def test_is_success_on_zero_exit_code(self):
        """ExecutionResult.is_success returns True when exit_code=0 and no timeout."""
        result = ExecutionResult(
            stdout="output",
            stderr="",
            exit_code=0,
            is_timeout=False,
        )
        assert result.is_success is True

    def test_is_success_false_on_nonzero_exit_code(self):
        """ExecutionResult.is_success returns False on non-zero exit."""
        result = ExecutionResult(
            stdout="",
            stderr="error message",
            exit_code=1,
            is_timeout=False,
        )
        assert result.is_success is False

    def test_is_success_false_on_timeout(self):
        """ExecutionResult.is_success returns False when timed out even with exit_code=0."""
        result = ExecutionResult(
            stdout="",
            stderr="timeout",
            exit_code=0,
            is_timeout=True,
        )
        assert result.is_success is False

    def test_is_success_false_on_timeout_with_nonzero_exit(self):
        """ExecutionResult.is_success returns False when both timeout and non-zero exit."""
        result = ExecutionResult(
            stdout="",
            stderr="timeout",
            exit_code=124,
            is_timeout=True,
        )
        assert result.is_success is False

    def test_is_success_false_on_various_exit_codes(self):
        """ExecutionResult.is_success returns False for various non-zero exit codes."""
        for exit_code in [1, 2, 5, 124, 137, 255]:
            result = ExecutionResult(
                stdout="",
                stderr="error",
                exit_code=exit_code,
                is_timeout=False,
            )
            assert result.is_success is False, f"Expected False for exit_code={exit_code}"

    def test_execution_result_is_frozen(self):
        """ExecutionResult is immutable (frozen dataclass)."""
        result = ExecutionResult(
            stdout="output",
            stderr="",
            exit_code=0,
            is_timeout=False,
        )
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[misc]


class TestAttempt:
    """Tests for Attempt dataclass and is_perfect property."""

    def _make_attempt(self, aggregated_score: float) -> Attempt:
        """Helper to create an Attempt with the given aggregated score."""
        example_result = ExampleResult(
            score=aggregated_score,
            error_type=ErrorType.NONE if aggregated_score >= 0.999 else ErrorType.MISSING_EXTRA,
            feedback="test feedback",
            actual_output="test",
            expected_output="test",
        )
        return Attempt(
            iteration=1,
            filter_code=".test",
            example_results=[example_result],
            aggregated_score=aggregated_score,
            primary_error=ErrorType.NONE if aggregated_score >= 0.999 else ErrorType.MISSING_EXTRA,
        )

    def test_is_perfect_at_exact_threshold(self):
        """Attempt.is_perfect returns True at exactly 0.999 threshold."""
        attempt = self._make_attempt(0.999)
        assert attempt.is_perfect is True

    def test_is_perfect_above_threshold(self):
        """Attempt.is_perfect returns True above 0.999 threshold."""
        attempt = self._make_attempt(1.0)
        assert attempt.is_perfect is True

    def test_is_perfect_slightly_above_threshold(self):
        """Attempt.is_perfect returns True for 0.9999."""
        attempt = self._make_attempt(0.9999)
        assert attempt.is_perfect is True

    def test_not_perfect_below_threshold(self):
        """Attempt.is_perfect returns False below 0.999 threshold."""
        attempt = self._make_attempt(0.998)
        assert attempt.is_perfect is False

    def test_not_perfect_at_zero(self):
        """Attempt.is_perfect returns False for score 0.0."""
        attempt = self._make_attempt(0.0)
        assert attempt.is_perfect is False

    def test_not_perfect_at_half(self):
        """Attempt.is_perfect returns False for score 0.5."""
        attempt = self._make_attempt(0.5)
        assert attempt.is_perfect is False

    def test_not_perfect_close_to_threshold(self):
        """Attempt.is_perfect returns False for 0.9989 (just below threshold)."""
        attempt = self._make_attempt(0.9989)
        assert attempt.is_perfect is False

    def test_attempt_is_frozen(self):
        """Attempt is immutable (frozen dataclass)."""
        attempt = self._make_attempt(1.0)
        with pytest.raises(AttributeError):
            attempt.aggregated_score = 0.5  # type: ignore[misc]


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_syntax_error_type_exists(self):
        """ErrorType.SYNTAX is defined."""
        assert ErrorType.SYNTAX.value == "syntax"

    def test_shape_error_type_exists(self):
        """ErrorType.SHAPE is defined."""
        assert ErrorType.SHAPE.value == "shape"

    def test_missing_extra_error_type_exists(self):
        """ErrorType.MISSING_EXTRA is defined."""
        assert ErrorType.MISSING_EXTRA.value == "missing_extra"

    def test_order_error_type_exists(self):
        """ErrorType.ORDER is defined."""
        assert ErrorType.ORDER.value == "order"

    def test_none_error_type_exists(self):
        """ErrorType.NONE is defined."""
        assert ErrorType.NONE.value == "none"

    def test_all_error_types_count(self):
        """ErrorType has exactly 5 values."""
        assert len(ErrorType) == 5

    def test_error_types_are_unique(self):
        """All ErrorType values are unique."""
        values = [e.value for e in ErrorType]
        assert len(values) == len(set(values))


class TestExample:
    """Tests for Example dataclass."""

    def test_example_creation(self):
        """Example can be created with input_data and expected_output."""
        example = Example(
            input_data={"key": "value"},
            expected_output="result",
        )
        assert example.input_data == {"key": "value"}
        assert example.expected_output == "result"

    def test_example_is_frozen(self):
        """Example is immutable (frozen dataclass)."""
        example = Example(input_data={}, expected_output=None)
        with pytest.raises(AttributeError):
            example.input_data = {"new": "value"}  # type: ignore[misc]


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Task can be created with all fields."""
        example = Example(input_data={"x": 1}, expected_output=1)
        task = Task(
            id="test-task",
            description="Test description",
            examples=[example],
        )
        assert task.id == "test-task"
        assert task.description == "Test description"
        assert len(task.examples) == 1

    def test_task_is_frozen(self):
        """Task is immutable (frozen dataclass)."""
        task = Task(id="test", description="test", examples=[])
        with pytest.raises(AttributeError):
            task.id = "new-id"  # type: ignore[misc]


class TestExampleResult:
    """Tests for ExampleResult dataclass."""

    def test_example_result_creation(self):
        """ExampleResult can be created with all fields."""
        result = ExampleResult(
            score=0.75,
            error_type=ErrorType.MISSING_EXTRA,
            feedback="Missing 2 elements",
            actual_output=[1, 2],
            expected_output=[1, 2, 3, 4],
        )
        assert result.score == 0.75
        assert result.error_type == ErrorType.MISSING_EXTRA
        assert result.feedback == "Missing 2 elements"

    def test_example_result_is_frozen(self):
        """ExampleResult is immutable (frozen dataclass)."""
        result = ExampleResult(
            score=1.0,
            error_type=ErrorType.NONE,
            feedback="",
            actual_output=None,
            expected_output=None,
        )
        with pytest.raises(AttributeError):
            result.score = 0.0  # type: ignore[misc]


class TestSolution:
    """Tests for Solution dataclass."""

    def test_solution_success(self):
        """Solution correctly represents a successful result."""
        solution = Solution(
            task_id="test-task",
            success=True,
            best_filter=".foo",
            best_score=1.0,
            iterations_used=2,
            history=[],
        )
        assert solution.success is True
        assert solution.best_filter == ".foo"
        assert solution.best_score == 1.0

    def test_solution_failure(self):
        """Solution correctly represents a failed result."""
        solution = Solution(
            task_id="test-task",
            success=False,
            best_filter=".partial",
            best_score=0.5,
            iterations_used=10,
            history=[],
        )
        assert solution.success is False
        assert solution.best_score == 0.5

    def test_solution_is_frozen(self):
        """Solution is immutable (frozen dataclass)."""
        solution = Solution(
            task_id="test",
            success=True,
            best_filter=".",
            best_score=1.0,
            iterations_used=1,
            history=[],
        )
        with pytest.raises(AttributeError):
            solution.success = False  # type: ignore[misc]
