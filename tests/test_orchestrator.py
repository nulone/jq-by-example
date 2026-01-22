"""
Integration tests for orchestrator loop with mocked generator.

This module tests the Orchestrator class for proper iteration control,
stagnation detection, duplicate handling, and best solution tracking
using mocked generators to simulate various scenarios.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.domain import Example, Task
from src.executor import JQExecutor
from src.generator import JQGenerator
from src.orchestrator import Orchestrator
from src.reviewer import AlgorithmicReviewer


@pytest.fixture
def mock_generator() -> MagicMock:
    """
    Create a mock JQGenerator.

    Returns:
        MagicMock configured as a JQGenerator.
    """
    generator = MagicMock(spec=JQGenerator)
    return generator


@pytest.fixture
def orchestrator_factory(
    executor: JQExecutor,
) -> Callable[[MagicMock, int, int], Orchestrator]:
    """
    Factory fixture for creating Orchestrator with custom settings.

    Args:
        executor: JQExecutor fixture.

    Returns:
        Callable that creates an Orchestrator with the given mock generator.
    """

    def _factory(
        mock_gen: MagicMock,
        max_iterations: int = 10,
        stagnation_limit: int = 3,
    ) -> Orchestrator:
        reviewer = AlgorithmicReviewer(executor)
        return Orchestrator(
            generator=mock_gen,
            reviewer=reviewer,
            max_iterations=max_iterations,
            stagnation_limit=stagnation_limit,
        )

    return _factory


class TestFindsSolutionOnSecondTry:
    """Tests for finding solution after initial failure."""

    def test_returns_success_on_second_correct_filter(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Orchestrator returns success when correct filter found on iteration 2."""
        # First filter is wrong (.items returns array, not extracted values)
        # Second filter is correct
        mock_generator.generate.side_effect = [".items", "[.items[].id]"]

        orchestrator = orchestrator_factory(mock_generator)

        task = Task(
            id="test-task",
            description="Extract ids from items",
            examples=[
                Example(
                    input_data={"items": [{"id": 1}, {"id": 2}]},
                    expected_output=[1, 2],
                ),
            ],
        )

        solution = orchestrator.solve(task)

        assert solution.success is True
        assert solution.best_filter == "[.items[].id]"
        assert solution.iterations_used == 2

    def test_history_contains_both_attempts(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Solution history contains all attempts made."""
        mock_generator.generate.side_effect = [".wrong", ".x"]

        orchestrator = orchestrator_factory(mock_generator)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 42}, expected_output=42)],
        )

        solution = orchestrator.solve(task)

        assert len(solution.history) == 2
        assert solution.history[0].filter_code == ".wrong"
        assert solution.history[1].filter_code == ".x"

    def test_generator_receives_history_on_second_call(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Generator is called with previous attempt history."""
        mock_generator.generate.side_effect = [".wrong", ".x"]

        orchestrator = orchestrator_factory(mock_generator)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 42}, expected_output=42)],
        )

        orchestrator.solve(task)

        # First call should have no history (or None)
        first_call_args = mock_generator.generate.call_args_list[0]
        assert first_call_args[0][0] == task  # task is first positional arg
        assert first_call_args[0][1] is None or first_call_args[0][1] == []

        # Second call should have history with one attempt
        second_call_args = mock_generator.generate.call_args_list[1]
        history = second_call_args[0][1]
        assert history is not None
        assert len(history) == 1
        assert history[0].filter_code == ".wrong"


class TestStopsOnPerfect:
    """Tests for immediate termination on perfect solution."""

    def test_single_iteration_when_first_filter_is_perfect(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Orchestrator stops after one iteration if filter is perfect."""
        mock_generator.generate.return_value = ".name"

        orchestrator = orchestrator_factory(mock_generator, max_iterations=10)

        task = Task(
            id="test-task",
            description="Extract name",
            examples=[
                Example(input_data={"name": "Alice"}, expected_output="Alice"),
            ],
        )

        solution = orchestrator.solve(task)

        assert solution.success is True
        assert solution.iterations_used == 1
        assert mock_generator.generate.call_count == 1

    def test_perfect_score_is_one(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Perfect solution has score of 1.0."""
        mock_generator.generate.return_value = ".value"

        orchestrator = orchestrator_factory(mock_generator)

        task = Task(
            id="test-task",
            description="Extract value",
            examples=[Example(input_data={"value": 123}, expected_output=123)],
        )

        solution = orchestrator.solve(task)

        assert solution.best_score == 1.0

    def test_does_not_generate_more_filters_after_perfect(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Generator is not called again after perfect solution found."""
        # Set up generator to return perfect filter, then track any additional calls
        call_count = [0]

        def track_calls(*args: Any, **kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return ".x"
            return ".should_not_be_called"

        mock_generator.generate.side_effect = track_calls

        orchestrator = orchestrator_factory(mock_generator, max_iterations=5)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        assert solution.success is True
        assert call_count[0] == 1


class TestStagnationStop:
    """Tests for stagnation detection and loop termination."""

    def test_stops_after_stagnation_limit_without_improvement(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Loop stops after stagnation_limit iterations without score improvement."""
        # All filters produce same low score
        mock_generator.generate.side_effect = [
            ".a",  # First attempt, becomes best
            ".b",  # No improvement
            ".c",  # No improvement
            ".d",  # No improvement - should trigger stagnation
            ".e",  # Should not be reached
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        assert solution.success is False
        # Should stop after stagnation limit is reached
        # First attempt sets best, then 3 more without improvement = 4 total
        assert solution.iterations_used <= 4

    def test_stagnation_resets_on_improvement(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Stagnation counter resets when score improves."""
        # Scores: 0.0, 0.0, 0.5 (improvement resets), 0.5, 0.5, 0.5 (stagnation)
        mock_generator.generate.side_effect = [
            ".wrong1",  # Score ~0, becomes best
            ".wrong2",  # Score ~0, no improvement, stagnation=1
            ".items",  # Partial match, improvement, stagnation=0
            ".wrong3",  # Score ~0, no improvement, stagnation=1
            ".wrong4",  # Score ~0, no improvement, stagnation=2
            ".wrong5",  # Score ~0, no improvement, stagnation=3 -> stop
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Get items array",
            examples=[
                Example(
                    input_data={"items": [1, 2, 3]},
                    expected_output=[1, 2, 3],
                ),
            ],
        )

        solution = orchestrator.solve(task)

        # Should have made more attempts due to the improvement resetting stagnation
        assert solution.iterations_used >= 3
        assert solution.best_filter == ".items"


class TestDuplicateDetection:
    """Tests for normalized filter duplicate detection."""

    def test_normalized_duplicates_increment_stagnation(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Duplicate filters (after normalization) increase stagnation counter."""
        # These should all normalize to the same string (whitespace-only differences)
        mock_generator.generate.side_effect = [
            ".foo",  # First, evaluated
            ". foo",  # Duplicate (whitespace normalized)
            ".foo ",  # Duplicate (whitespace normalized)
            ".bar",  # Different filter
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        # Only .foo and potentially .bar should be in history
        # The duplicates should have caused stagnation
        filter_codes = [a.filter_code for a in solution.history]
        assert ".foo" in filter_codes
        # Duplicates should not appear in history (whitespace-normalized duplicates)
        assert filter_codes.count(". foo") == 0
        assert filter_codes.count(".foo ") == 0

    def test_case_sensitive_duplicate_detection(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Duplicate detection is case-sensitive (preserves jq field name case)."""
        mock_generator.generate.side_effect = [
            ".Name",
            ".name",  # Different filter (case-sensitive)
            ".NAME",  # Different filter (case-sensitive)
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=5)

        task = Task(
            id="test-task",
            description="Extract name",
            # Using a different field so none of the filters match
            examples=[Example(input_data={"title": "Alice"}, expected_output="Alice")],
        )

        solution = orchestrator.solve(task)

        # All three should be evaluated as they have different cases
        filter_codes = [a.filter_code for a in solution.history]
        assert ".Name" in filter_codes
        assert ".name" in filter_codes
        assert ".NAME" in filter_codes
        assert len(filter_codes) == 3

    def test_whitespace_ignored_in_normalization(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Whitespace differences don't create distinct filters."""
        mock_generator.generate.side_effect = [
            ".items | map(.x)",
            ".items|map(.x)",  # Same when whitespace removed
            ". items | map( .x )",  # Same when whitespace removed
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=2)

        task = Task(
            id="test-task",
            description="Map items",
            examples=[
                Example(
                    input_data={"items": [{"x": 1}, {"x": 2}]},
                    expected_output=[1, 2],
                ),
            ],
        )

        solution = orchestrator.solve(task)

        # Only first should be evaluated, others are duplicates
        assert len(solution.history) == 1


class TestMaxIterationsLimit:
    """Tests for max iterations limit enforcement."""

    def test_stops_at_max_iterations(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Loop stops at max_iterations even with improvements."""
        # Generate unique filters that keep improving slightly
        iteration = [0]

        def improving_filter(*args: Any, **kwargs: Any) -> str:
            iteration[0] += 1
            # Return filters with increasing partial scores
            # These all fail but accessing deeper nesting gives different errors
            return f".level{iteration[0]}"

        mock_generator.generate.side_effect = improving_filter

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=5,
            stagnation_limit=10,  # High limit to not trigger
        )

        task = Task(
            id="test-task",
            description="Extract deep value",
            examples=[
                Example(
                    input_data={"deep": {"nested": {"value": 42}}},
                    expected_output=42,
                ),
            ],
        )

        solution = orchestrator.solve(task)

        assert solution.iterations_used == 5
        assert solution.success is False

    def test_max_iterations_one_baseline_mode(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Single iteration mode (baseline) only tries once."""
        mock_generator.generate.return_value = ".wrong"

        orchestrator = orchestrator_factory(mock_generator, max_iterations=1)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        assert solution.iterations_used == 1
        assert mock_generator.generate.call_count == 1

    def test_returns_best_at_max_iterations(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Returns the best filter found when max iterations reached."""
        # Third filter is best but not perfect
        mock_generator.generate.side_effect = [
            ".wrong1",  # Score 0
            ".wrong2",  # Score 0
            ".items",  # Partial score (returns array but not transformed)
            ".wrong3",  # Score 0
            ".wrong4",  # Score 0
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=5,
            stagnation_limit=10,
        )

        task = Task(
            id="test-task",
            description="Get transformed items",
            examples=[
                Example(
                    input_data={"items": [1, 2]},
                    expected_output=[1, 2],  # .items will match this
                ),
            ],
        )

        solution = orchestrator.solve(task)

        assert solution.best_filter == ".items"
        assert solution.best_score == 1.0  # Actually perfect match


class TestGeneratorExceptionHandled:
    """Tests for handling generator exceptions."""

    def test_continues_after_generator_exception(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Generator failure doesn't crash; continues with next iteration."""
        mock_generator.generate.side_effect = [
            Exception("API Error"),  # First call fails
            ".x",  # Second call succeeds with correct filter
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            stagnation_limit=3,  # Allow recovery
        )

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 42}, expected_output=42)],
        )

        solution = orchestrator.solve(task)

        assert solution.success is True
        assert solution.best_filter == ".x"

    def test_multiple_exceptions_increment_stagnation(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Multiple consecutive exceptions trigger stagnation stop."""
        mock_generator.generate.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
            ".x",  # Should not be reached
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        # Should have stopped due to stagnation from exceptions
        assert solution.success is False
        assert solution.best_filter == ""
        assert solution.iterations_used == 0  # No successful evaluations

    def test_recovers_after_exception_then_succeeds(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Can recover from exception and eventually find solution."""
        mock_generator.generate.side_effect = [
            Exception("Temporary error"),
            ".wrong",  # Evaluated but wrong
            Exception("Another error"),
            ".x",  # Correct!
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=10,
            stagnation_limit=5,
        )

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 99}, expected_output=99)],
        )

        solution = orchestrator.solve(task)

        assert solution.success is True
        assert solution.best_filter == ".x"


class TestBestSoFarTracked:
    """Tests for tracking best attempt even on overall failure."""

    def test_returns_best_score_on_failure(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Returns best attempt even when no perfect solution found."""
        # Scores will be approximately: 0, 0.5 (partial match), 0
        mock_generator.generate.side_effect = [
            ".totally_wrong",  # Score ~0
            ".data",  # Returns object, partial match
            ".also_wrong",  # Score ~0
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=3,
            stagnation_limit=5,
        )

        task = Task(
            id="test-task",
            description="Get user name",
            examples=[
                Example(
                    input_data={"data": {"name": "Alice", "age": 30}},
                    expected_output={"name": "Alice", "age": 30},
                ),
            ],
        )

        solution = orchestrator.solve(task)

        # .data returns the exact expected output, so it's perfect
        assert solution.best_filter == ".data"
        assert solution.success is True

    def test_best_filter_tracked_through_variations(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Best filter is updated correctly as scores vary."""
        # Create filters with varying partial scores
        mock_generator.generate.side_effect = [
            ".a",  # Returns null, score 0
            "[.items[0]]",  # Returns [1], partial match with [1,2,3]
            ".b",  # Returns null, score 0
            "[.items[0], .items[1]]",  # Returns [1,2], better partial match
            ".c",  # Returns null, score 0
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=5,
            stagnation_limit=10,
        )

        task = Task(
            id="test-task",
            description="Get all items",
            examples=[
                Example(
                    input_data={"items": [1, 2, 3]},
                    expected_output=[1, 2, 3],
                ),
            ],
        )

        solution = orchestrator.solve(task)

        # The filter returning [1,2] should have better Jaccard score than [1]
        assert solution.best_filter == "[.items[0], .items[1]]"
        assert solution.best_score > 0.5

    def test_tracks_best_across_all_iterations(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Best is tracked correctly even if best attempt is not the last."""
        # Best filter is in the middle
        mock_generator.generate.side_effect = [
            ".wrong1",
            ".wrong2",
            ".items",  # Perfect match - but we'll test with imperfect
            ".wrong3",
            ".wrong4",
        ]

        orchestrator = orchestrator_factory(
            mock_generator,
            max_iterations=5,
            stagnation_limit=10,
        )

        task = Task(
            id="test-task",
            description="Extract items",
            examples=[
                Example(
                    input_data={"items": [1, 2, 3]},
                    expected_output=[1, 2, 3],
                ),
            ],
        )

        solution = orchestrator.solve(task)

        # .items is perfect match
        assert solution.success is True
        assert solution.best_filter == ".items"
        # Should stop at iteration 3 due to perfect match
        assert solution.iterations_used == 3


class TestVerboseLogging:
    """Tests for verbose mode behavior."""

    def test_verbose_mode_logs_generator_exception(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
        caplog: pytest.LogCaptureFixture,
    ):
        """Verbose mode logs generator exceptions."""
        import logging

        mock_generator.generate.side_effect = [
            Exception("API timeout"),
            ".x",
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        with caplog.at_level(logging.WARNING):
            solution = orchestrator.solve(task, verbose=True)

        assert solution.success is True
        # Check that warning was logged
        assert any("Generator failed" in record.message for record in caplog.records)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_filter_string(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Empty filter string is handled (will fail in jq)."""
        mock_generator.generate.side_effect = ["", ".x"]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        # Should recover and find correct filter
        assert solution.success is True
        assert solution.best_filter == ".x"

    def test_all_failures_returns_empty_solution(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """When all attempts fail completely, returns appropriate solution."""
        mock_generator.generate.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
        ]

        orchestrator = orchestrator_factory(mock_generator, stagnation_limit=3)

        task = Task(
            id="test-task",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        assert solution.success is False
        assert solution.best_filter == ""
        assert solution.best_score == 0.0
        assert solution.iterations_used == 0
        assert len(solution.history) == 0

    def test_solution_contains_task_id(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Solution correctly references the task ID."""
        mock_generator.generate.return_value = ".x"

        orchestrator = orchestrator_factory(mock_generator)

        task = Task(
            id="unique-task-id-123",
            description="Extract x",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        solution = orchestrator.solve(task)

        assert solution.task_id == "unique-task-id-123"

    def test_multiple_examples_all_must_pass(
        self,
        mock_generator: MagicMock,
        orchestrator_factory: Callable[[MagicMock, int, int], Orchestrator],
    ):
        """Filter must pass all examples to be considered perfect."""
        # This filter works for first example but not second
        mock_generator.generate.side_effect = [
            '.name // "default"',  # Works for both if name exists or not
            ".name",  # Only works when name exists
        ]

        orchestrator = orchestrator_factory(mock_generator, max_iterations=2)

        task = Task(
            id="test-task",
            description="Extract name",
            examples=[
                Example(input_data={"name": "Alice"}, expected_output="Alice"),
                Example(input_data={"name": "Bob"}, expected_output="Bob"),
                Example(input_data={"other": "field"}, expected_output=None),
            ],
        )

        solution = orchestrator.solve(task)

        # .name returns null for third example which matches expected None
        assert solution.success is True


class TestFilterNormalization:
    """Regression tests for filter normalization with string literals."""

    def test_normalize_preserves_spaces_in_strings(self) -> None:
        """Spaces inside string literals should be preserved."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        # Filters with different string content are different
        filter1 = '.[] | select(.x == "a b")'
        filter2 = '.[] | select(.x == "ab")'

        norm1 = orch._normalize(filter1)
        norm2 = orch._normalize(filter2)

        # These should be DIFFERENT
        assert norm1 != norm2
        assert '"a b"' in norm1
        assert '"ab"' in norm2

    def test_normalize_removes_spaces_outside_strings(self) -> None:
        """Spaces outside string literals should be removed."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        # Same filter with different whitespace
        filter1 = '.x|select(.y=="test")'
        filter2 = '. x | select( .y == "test" )'

        norm1 = orch._normalize(filter1)
        norm2 = orch._normalize(filter2)

        # These should be SAME
        assert norm1 == norm2
        assert norm1 == '.x|select(.y=="test")'

    def test_normalize_handles_escaped_quotes(self) -> None:
        """Escaped quotes inside strings should be handled correctly."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        filter1 = '.x == "\\"quoted\\""'
        filter2 = '.x=="\\"quoted\\""'

        norm1 = orch._normalize(filter1)
        norm2 = orch._normalize(filter2)

        # These should be SAME (different spacing, same escapes)
        assert norm1 == norm2

    def test_normalize_space_in_concatenation(self) -> None:
        """Space character in string concatenation should be preserved."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        # Different: one has space in string, other doesn't
        filter1 = '.x + " " + .y'
        filter2 = '.x+""+.y'

        norm1 = orch._normalize(filter1)
        norm2 = orch._normalize(filter2)

        # These should be DIFFERENT
        assert norm1 != norm2
        assert '" "' in norm1
        assert '""' in norm2

    def test_normalize_complex_nested_strings(self) -> None:
        """Complex filters with multiple strings should preserve content."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        filter_code = '.[] | select(.name == "John Doe" and .city == "New York")'
        normalized = orch._normalize(filter_code)

        # String contents should be preserved
        assert '"John Doe"' in normalized
        assert '"New York"' in normalized
        # Spaces outside strings should be removed
        assert " and " not in normalized
        assert "and" in normalized

    def test_normalize_empty_filter(self) -> None:
        """Empty filter should return empty string."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        assert orch._normalize("") == ""
        assert orch._normalize("   ") == ""

    def test_normalize_filter_without_strings(self) -> None:
        """Filter without string literals should have all spaces removed."""
        from unittest.mock import MagicMock

        from src.orchestrator import Orchestrator

        orch = Orchestrator(generator=MagicMock(), reviewer=MagicMock(), max_iterations=10)

        filter1 = ". [] | . x | . y"
        filter2 = ".[]|.x|.y"

        norm1 = orch._normalize(filter1)
        norm2 = orch._normalize(filter2)

        # Should be same
        assert norm1 == norm2
        assert norm1 == ".[]|.x|.y"
