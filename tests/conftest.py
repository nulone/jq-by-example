"""
Shared pytest fixtures for all JQ-Synth tests.

This module provides common fixtures used across the test suite, including
executor instances, reviewer instances, and task factory helpers.
"""

from collections.abc import Callable
from typing import Any

import pytest

from src.domain import Example, Task
from src.executor import JQExecutor
from src.reviewer import AlgorithmicReviewer


@pytest.fixture
def executor() -> JQExecutor:
    """
    Create a default JQExecutor instance.

    Returns:
        JQExecutor with default settings (1s timeout, 1MB output limit).

    Raises:
        pytest.skip: If jq binary is not installed on the system.
    """
    try:
        return JQExecutor()
    except RuntimeError as e:
        pytest.skip(f"jq binary not available: {e}")


@pytest.fixture
def reviewer(executor: JQExecutor) -> AlgorithmicReviewer:
    """
    Create an AlgorithmicReviewer instance with the default executor.

    Args:
        executor: JQExecutor fixture instance.

    Returns:
        AlgorithmicReviewer configured with the provided executor.
    """
    return AlgorithmicReviewer(executor)


@pytest.fixture
def make_task() -> Callable[[Any, Any, str], Task]:
    """
    Factory fixture for creating simple single-example Tasks.

    Returns:
        A callable that creates a Task with one example.

    Example:
        def test_something(make_task):
            task = make_task({"x": 1}, 1)
            # task has id="test-task", one example with input {"x": 1}, expected 1
    """

    def _make_task(
        input_data: Any,
        expected_output: Any,
        description: str = "Test task",
    ) -> Task:
        """
        Create a Task with a single example.

        Args:
            input_data: The JSON input for the example.
            expected_output: The expected JSON output for the example.
            description: Optional task description. Defaults to "Test task".

        Returns:
            Task with id="test-task" and a single example.
        """
        example = Example(input_data=input_data, expected_output=expected_output)
        return Task(
            id="test-task",
            description=description,
            examples=[example],
        )

    return _make_task
