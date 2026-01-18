"""
Domain data structures and contracts for the JQ-Synth pipeline.

This module defines all immutable data structures used throughout the pipeline,
including error types, task definitions, execution results, and solution tracking.
All classes are frozen dataclasses to ensure immutability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorType(Enum):
    """
    Classification of errors encountered during jq filter evaluation.

    Priority order (highest to lowest): SYNTAX > SHAPE > MISSING_EXTRA > ORDER > NONE.
    This ordering is used to determine the primary error when multiple error types
    are present across examples.
    """

    SYNTAX = "syntax"
    SHAPE = "shape"
    MISSING_EXTRA = "missing_extra"
    ORDER = "order"
    NONE = "none"


@dataclass(frozen=True)
class Example:
    """
    A single input/output example for a jq synthesis task.

    Attributes:
        input_data: The JSON input to be processed by the jq filter.
        expected_output: The expected JSON output after applying the filter.
    """

    input_data: Any
    expected_output: Any


@dataclass(frozen=True)
class Task:
    """
    A jq synthesis task with description and examples.

    Attributes:
        id: Unique identifier for the task.
        description: Human-readable description of what the filter should do.
        examples: List of input/output examples demonstrating expected behavior.
    """

    id: str
    description: str
    examples: list[Example]


@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of executing a jq filter via subprocess.

    Attributes:
        stdout: Standard output from jq execution.
        stderr: Standard error from jq execution.
        exit_code: Process exit code (0 for success, non-zero for errors).
        is_timeout: Whether the execution timed out.
    """

    stdout: str
    stderr: str
    exit_code: int
    is_timeout: bool

    @property
    def is_success(self) -> bool:
        """
        Check if the execution was successful.

        Returns:
            True if exit code is 0 and execution did not timeout, False otherwise.
        """
        return self.exit_code == 0 and not self.is_timeout


@dataclass(frozen=True)
class ExampleResult:
    """
    Result of evaluating a jq filter against a single example.

    Attributes:
        score: Similarity score between 0.0 and 1.0 (1.0 = perfect match).
        error_type: Classification of the error, if any.
        feedback: Human-readable description of the issue for LLM feedback.
        actual_output: The actual output produced by the jq filter.
        expected_output: The expected output from the example.
    """

    score: float
    error_type: ErrorType
    feedback: str
    actual_output: Any
    expected_output: Any


@dataclass(frozen=True)
class Attempt:
    """
    A single attempt at solving a jq synthesis task.

    Attributes:
        iteration: The iteration number (1-indexed).
        filter_code: The jq filter code that was tried.
        example_results: Results for each example in the task.
        aggregated_score: Average score across all examples.
        primary_error: The most significant error type encountered.
    """

    iteration: int
    filter_code: str
    example_results: list[ExampleResult]
    aggregated_score: float
    primary_error: ErrorType

    @property
    def is_perfect(self) -> bool:
        """
        Check if this attempt achieved a perfect score.

        Uses a threshold of 0.999 instead of 1.0 to handle floating-point
        precision issues in score calculations.

        Returns:
            True if aggregated_score >= 0.999, False otherwise.
        """
        return self.aggregated_score >= 0.999


@dataclass(frozen=True)
class Solution:
    """
    Final solution for a jq synthesis task.

    Attributes:
        task_id: The ID of the task that was solved.
        success: Whether a perfect solution was found.
        best_filter: The best jq filter found during solving.
        best_score: The score of the best filter.
        iterations_used: Total number of iterations attempted.
        history: List of all attempts made during solving.
    """

    task_id: str
    success: bool
    best_filter: str
    best_score: float
    iterations_used: int
    history: list[Attempt]
