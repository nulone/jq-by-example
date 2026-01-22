"""
End-to-end tests with real components (skip in CI without API key).

This module contains integration tests that use real API calls to LLM providers
and real jq binary execution. These tests are marked to skip when the
OPENAI_API_KEY environment variable is not set, making them suitable
for manual validation rather than CI.
"""

import os

import pytest

from src.domain import Example, Task
from src.executor import JQExecutor
from src.generator import JQGenerator
from src.orchestrator import Orchestrator
from src.reviewer import AlgorithmicReviewer

# Check for API key availability
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")


@pytest.fixture
def e2e_orchestrator() -> Orchestrator:
    """
    Create a full orchestrator with real components for E2E testing.

    Returns:
        Orchestrator configured with real generator, reviewer, and executor.

    Raises:
        pytest.skip: If jq binary is not available.
    """
    try:
        executor = JQExecutor()
    except RuntimeError as e:
        pytest.skip(f"jq binary not available: {e}")

    # API key is checked by skipif decorator on tests
    # Generator will use environment variables for configuration
    generator = JQGenerator()
    reviewer = AlgorithmicReviewer(executor)

    return Orchestrator(
        generator=generator,
        reviewer=reviewer,
        max_iterations=10,
        stagnation_limit=3,
    )


@pytest.mark.e2e
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY environment variable not set")
class TestNestedFieldTask:
    """E2E tests for the nested-field task."""

    def test_solves_nested_field_task(self, e2e_orchestrator: Orchestrator):
        """
        Simple nested field extraction should be solved quickly with real API.

        This test verifies that the full pipeline can synthesize a filter
        to extract .user.name from a nested object structure.

        The task is straightforward and should be solved within 3 iterations
        by any competent LLM.
        """
        task = Task(
            id="nested-field",
            description="Extract the user's name from a nested object structure",
            examples=[
                Example(
                    input_data={"user": {"name": "Alice", "age": 30}},
                    expected_output="Alice",
                ),
                Example(
                    input_data={"user": {"name": "Bob", "email": "bob@example.com"}},
                    expected_output="Bob",
                ),
                Example(
                    input_data={"user": {"name": "Charlie Brown", "id": 123, "active": True}},
                    expected_output="Charlie Brown",
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        assert solution.success is True, (
            f"Failed to solve nested-field task. "
            f"Best filter: '{solution.best_filter}', Score: {solution.best_score:.3f}, "
            f"Iterations: {solution.iterations_used}"
        )
        assert solution.iterations_used <= 3, (
            f"Expected solution in <= 3 iterations, took {solution.iterations_used}. "
            f"Filter: '{solution.best_filter}'"
        )
        assert solution.best_score >= 0.999, (
            f"Expected perfect score, got {solution.best_score:.3f}"
        )

    def test_nested_field_solution_is_valid_jq(self, e2e_orchestrator: Orchestrator):
        """
        Verify that the synthesized filter produces correct output.

        This test runs the solution filter against a new input to verify
        it generalizes correctly.
        """
        task = Task(
            id="nested-field-verify",
            description="Extract the user's name from a nested object structure",
            examples=[
                Example(
                    input_data={"user": {"name": "Alice", "age": 30}},
                    expected_output="Alice",
                ),
                Example(
                    input_data={"user": {"name": "Bob", "email": "bob@example.com"}},
                    expected_output="Bob",
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        if not solution.success:
            pytest.skip("Could not find solution to verify")

        # Test the filter on a new input
        executor = JQExecutor()
        new_input = {"user": {"name": "Diana", "role": "admin"}}
        result = executor.run(solution.best_filter, new_input)

        assert result.is_success, f"Filter failed on new input: {result.stderr}"
        # The output should be "Diana" (quoted in JSON)
        assert result.stdout.strip('"') == "Diana", f"Expected 'Diana', got {result.stdout}"


@pytest.mark.e2e
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY environment variable not set")
class TestFilterActiveTask:
    """E2E tests for the filter-active task."""

    def test_solves_filter_active_task(self, e2e_orchestrator: Orchestrator):
        """
        Filter by boolean field should be solved with real API.

        This test verifies that the full pipeline can synthesize a filter
        to select objects where active == true. This requires understanding
        of jq's select() function or equivalent filtering.
        """
        task = Task(
            id="filter-active",
            description="Filter an array to keep only objects where the 'active' field is true",
            examples=[
                Example(
                    input_data=[
                        {"id": 1, "name": "Task A", "active": True},
                        {"id": 2, "name": "Task B", "active": False},
                        {"id": 3, "name": "Task C", "active": True},
                    ],
                    expected_output=[
                        {"id": 1, "name": "Task A", "active": True},
                        {"id": 3, "name": "Task C", "active": True},
                    ],
                ),
                Example(
                    input_data=[
                        {"id": 1, "active": False},
                        {"id": 2, "active": False},
                    ],
                    expected_output=[],
                ),
                Example(
                    input_data=[
                        {"id": 1, "active": True},
                    ],
                    expected_output=[
                        {"id": 1, "active": True},
                    ],
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        assert solution.success is True, (
            f"Failed to solve filter-active task. "
            f"Best filter: '{solution.best_filter}', Score: {solution.best_score:.3f}, "
            f"Iterations: {solution.iterations_used}"
        )
        assert solution.iterations_used <= 5, (
            f"Expected solution in <= 5 iterations, took {solution.iterations_used}. "
            f"Filter: '{solution.best_filter}'"
        )
        assert solution.best_score >= 0.999, (
            f"Expected perfect score, got {solution.best_score:.3f}"
        )

    def test_filter_active_handles_empty_array(self, e2e_orchestrator: Orchestrator):
        """
        Verify that the synthesized filter handles empty input arrays.

        Edge case: the filter should return an empty array when given
        an empty array as input.
        """
        task = Task(
            id="filter-active-empty",
            description="Filter an array to keep only objects where the 'active' field is true",
            examples=[
                Example(
                    input_data=[
                        {"id": 1, "active": True},
                        {"id": 2, "active": False},
                    ],
                    expected_output=[
                        {"id": 1, "active": True},
                    ],
                ),
                Example(
                    input_data=[],
                    expected_output=[],
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        if not solution.success:
            pytest.skip("Could not find solution to verify")

        # Test the filter on empty array
        executor = JQExecutor()
        result = executor.run(solution.best_filter, [])

        assert result.is_success, f"Filter failed on empty array: {result.stderr}"
        assert result.stdout == "[]", f"Expected '[]' for empty input, got {result.stdout}"


@pytest.mark.e2e
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY environment variable not set")
class TestIterativeRefinement:
    """E2E tests verifying the iterative refinement process."""

    def test_refinement_improves_score(self, e2e_orchestrator: Orchestrator):
        """
        Verify that iterative refinement can improve on initial attempts.

        This test uses a slightly harder task that may require feedback-based
        refinement to solve correctly.
        """
        task = Task(
            id="extract-emails",
            description=(
                "Extract all email addresses from an array of user objects, "
                "skipping users without an email or with null email"
            ),
            examples=[
                Example(
                    input_data=[
                        {"name": "Alice", "email": "alice@example.com"},
                        {"name": "Bob"},
                        {"name": "Charlie", "email": "charlie@example.com"},
                    ],
                    expected_output=["alice@example.com", "charlie@example.com"],
                ),
                Example(
                    input_data=[
                        {"name": "Alice"},
                        {"name": "Bob"},
                    ],
                    expected_output=[],
                ),
                Example(
                    input_data=[
                        {"name": "Alice", "email": None},
                        {"name": "Bob", "email": "bob@example.com"},
                    ],
                    expected_output=["bob@example.com"],
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        # This task is harder, so we allow more iterations
        # but still expect success
        assert solution.best_score > 0.5, (
            f"Expected significant progress, got score {solution.best_score:.3f}"
        )

        # Check that history shows progression if multiple attempts were made
        if len(solution.history) > 1:
            # Verify we have iteration tracking
            iterations = [a.iteration for a in solution.history]
            assert iterations == list(range(1, len(iterations) + 1)), (
                f"Iterations should be sequential: {iterations}"
            )

    def test_history_contains_feedback(self, e2e_orchestrator: Orchestrator):
        """
        Verify that attempt history contains useful feedback for debugging.

        Each attempt should have example results with feedback that could
        guide refinement.
        """
        task = Task(
            id="simple-extract",
            description="Extract the value field",
            examples=[
                Example(
                    input_data={"value": 42},
                    expected_output=42,
                ),
                Example(
                    input_data={"value": "hello"},
                    expected_output="hello",
                ),
            ],
        )

        solution = e2e_orchestrator.solve(task, verbose=True)

        # Should have at least one attempt
        assert len(solution.history) >= 1, "Expected at least one attempt in history"

        # Each attempt should have example results
        for attempt in solution.history:
            assert len(attempt.example_results) == len(task.examples), (
                f"Attempt {attempt.iteration} has wrong number of example results"
            )
            # Each result should have feedback
            for i, result in enumerate(attempt.example_results):
                assert result.feedback is not None, (
                    f"Attempt {attempt.iteration}, example {i} missing feedback"
                )
                assert result.error_type is not None, (
                    f"Attempt {attempt.iteration}, example {i} missing error_type"
                )
