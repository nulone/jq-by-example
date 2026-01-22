"""
Algorithmic diagnosis of jq output - classifies errors and calculates scores without LLM.

This module provides the AlgorithmicReviewer class that evaluates jq filter outputs
against expected results, calculating similarity scores and classifying error types
to provide actionable feedback for the LLM generator.
"""

import json
import logging
from collections import Counter
from typing import Any, ClassVar

from src.domain import Attempt, ErrorType, ExampleResult, Task
from src.executor import ExecutionResult, JQExecutor

logger = logging.getLogger(__name__)


# Sentinel to distinguish parse failure from valid None/null
class _ParseError:
    pass


_PARSE_ERROR = _ParseError()


class AlgorithmicReviewer:
    """
    Evaluates jq filters against task examples using algorithmic diagnosis.

    This class runs filters through the executor and analyzes outputs to:
    - Calculate similarity scores (0.0 to 1.0)
    - Classify error types (SYNTAX, SHAPE, MISSING_EXTRA, ORDER, NONE)
    - Generate human-readable feedback for LLM refinement

    Attributes:
        executor: The JQExecutor instance used to run filters.
    """

    # Error type priority for selecting primary error (higher index = higher priority)
    _ERROR_PRIORITY: ClassVar[dict[ErrorType, int]] = {
        ErrorType.NONE: 0,
        ErrorType.ORDER: 1,
        ErrorType.MISSING_EXTRA: 2,
        ErrorType.SHAPE: 3,
        ErrorType.SYNTAX: 4,
    }

    def __init__(self, executor: JQExecutor) -> None:
        """
        Initialize the algorithmic reviewer.

        Args:
            executor: JQExecutor instance for running jq filters.
        """
        self.executor = executor
        logger.debug("AlgorithmicReviewer initialized")

    def evaluate(self, task: Task, filter_code: str) -> Attempt:
        """
        Evaluate a jq filter against all examples in a task.

        Args:
            task: The task containing examples to evaluate against.
            filter_code: The jq filter expression to evaluate.

        Returns:
            Attempt containing results for each example, aggregated score,
            and primary error type.
        """
        logger.info("Evaluating filter '%s' against task '%s'", filter_code, task.id)

        example_results: list[ExampleResult] = []

        for i, example in enumerate(task.examples):
            exec_result = self.executor.run(filter_code, example.input_data)
            result = self._diagnose(exec_result, example.expected_output)
            example_results.append(result)

            logger.debug(
                "Example %d: score=%.3f, error_type=%s",
                i + 1,
                result.score,
                result.error_type.value,
            )

        # Calculate aggregated score (average)
        if example_results:
            aggregated_score = sum(r.score for r in example_results) / len(example_results)
        else:
            aggregated_score = 0.0

        # Determine primary error
        primary_error = self._primary_error(example_results)

        # Create attempt (iteration=0 as placeholder, orchestrator will set it)
        attempt = Attempt(
            iteration=0,
            filter_code=filter_code,
            example_results=example_results,
            aggregated_score=aggregated_score,
            primary_error=primary_error,
        )

        logger.info(
            "Evaluation complete: aggregated_score=%.3f, primary_error=%s, is_perfect=%s",
            aggregated_score,
            primary_error.value,
            attempt.is_perfect,
        )

        return attempt

    def _diagnose(self, exec_result: ExecutionResult, expected: Any) -> ExampleResult:
        """
        Diagnose a single execution result against expected output.

        Args:
            exec_result: The result from executing the jq filter.
            expected: The expected output value.

        Returns:
            ExampleResult with score, error type, and feedback.
        """
        # Handle execution failures
        if exec_result.is_timeout:
            return ExampleResult(
                score=0.0,
                error_type=ErrorType.SYNTAX,
                feedback="Filter execution timed out - possible infinite loop",
                actual_output=None,
                expected_output=expected,
            )

        if not exec_result.is_success:
            return ExampleResult(
                score=0.0,
                error_type=ErrorType.SYNTAX,
                feedback=f"jq error: {exec_result.stderr}",
                actual_output=exec_result.stdout,
                expected_output=expected,
            )

        # Try to parse the output as JSON
        actual = self._parse_jq_output(exec_result.stdout)

        if actual is _PARSE_ERROR:
            return ExampleResult(
                score=0.0,
                error_type=ErrorType.SYNTAX,
                feedback=f"Output is not valid JSON: {exec_result.stdout[:100]}",
                actual_output=exec_result.stdout,
                expected_output=expected,
            )

        # Analyze the parsed output against expected
        score, error_type, feedback = self._analyze(actual, expected)

        return ExampleResult(
            score=score,
            error_type=error_type,
            feedback=feedback,
            actual_output=actual,
            expected_output=expected,
        )

    def _parse_jq_output(self, stdout: str) -> Any:
        """
        Parse jq output, handling both single values and multi-line output.

        Args:
            stdout: The stdout from jq execution.

        Returns:
            Parsed JSON value, or _PARSE_ERROR sentinel if parsing fails.

        Note:
            Empty stdout is treated as a parse error because jq filters like 'empty'
            produce no output, which is semantically different from outputting 'null'.
            A filter that outputs null will produce stdout='null', which parses to None.
        """
        stdout = stdout.strip()

        # Empty output is an error - jq should output at least 'null' if that's intended
        # Filters like 'empty' or 'select(false)' produce no output, which should fail
        if not stdout:
            return _PARSE_ERROR

        # Try single JSON value first
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

        # Handle multi-line output (e.g., from .[].x producing multiple values)
        lines = stdout.split("\n")
        if len(lines) > 1:
            try:
                return [json.loads(line) for line in lines if line.strip()]
            except json.JSONDecodeError:
                return _PARSE_ERROR

        return _PARSE_ERROR

    def _analyze(self, actual: Any, expected: Any) -> tuple[float, ErrorType, str]:
        """
        Analyze actual output against expected output.

        Args:
            actual: The actual parsed output.
            expected: The expected output.

        Returns:
            Tuple of (score, error_type, feedback).
        """
        # Perfect match
        if actual == expected:
            return 1.0, ErrorType.NONE, "Perfect match"

        # Type mismatch (shape error)
        actual_type = type(actual).__name__
        expected_type = type(expected).__name__

        if isinstance(expected, list) and not isinstance(actual, list):
            return 0.0, ErrorType.SHAPE, f"Expected list but got {actual_type}"

        if isinstance(expected, dict) and not isinstance(actual, dict):
            return 0.0, ErrorType.SHAPE, f"Expected dict but got {actual_type}"

        if isinstance(actual, list) and not isinstance(expected, list):
            return 0.0, ErrorType.SHAPE, f"Expected {expected_type} but got list"

        if isinstance(actual, dict) and not isinstance(expected, dict):
            return 0.0, ErrorType.SHAPE, f"Expected {expected_type} but got dict"

        # List analysis
        if isinstance(expected, list) and isinstance(actual, list):
            return self._analyze_list(actual, expected)

        # Dict analysis
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._analyze_dict(actual, expected)

        # Scalar comparison (same type but different value)
        if type(actual) is type(expected):
            return 0.0, ErrorType.MISSING_EXTRA, f"Expected {expected!r} but got {actual!r}"

        # Type mismatch for scalars
        return 0.0, ErrorType.SHAPE, f"Type mismatch: expected {expected_type}, got {actual_type}"

    def _analyze_list(self, actual: list[Any], expected: list[Any]) -> tuple[float, ErrorType, str]:
        """
        Analyze list outputs using Jaccard similarity and order detection.

        Args:
            actual: The actual list output.
            expected: The expected list output.

        Returns:
            Tuple of (score, error_type, feedback).
        """
        # Handle empty lists
        if not expected and not actual:
            return 1.0, ErrorType.NONE, "Perfect match (both empty)"

        if not expected:
            return 0.0, ErrorType.MISSING_EXTRA, f"Expected empty list but got {len(actual)} items"

        if not actual:
            return (
                0.0,
                ErrorType.MISSING_EXTRA,
                f"Expected {len(expected)} items but got empty list",
            )

        # Convert to comparable strings for multiset operations
        def to_str(item: Any) -> str:
            return json.dumps(item, sort_keys=True)

        actual_strs = [to_str(item) for item in actual]
        expected_strs = [to_str(item) for item in expected]

        actual_counter = Counter(actual_strs)
        expected_counter = Counter(expected_strs)

        # Check for perfect match first
        if actual_strs == expected_strs:
            return 1.0, ErrorType.NONE, "Perfect match"

        # Check for order issues (same elements, different order)
        sorted_actual = sorted(actual_strs)
        sorted_expected = sorted(expected_strs)

        if sorted_actual == sorted_expected:
            # Same elements but wrong order
            return 0.8, ErrorType.ORDER, "Correct elements but wrong order"

        # Calculate Jaccard similarity for multisets
        # Intersection: sum of min counts, Union: sum of max counts
        all_elements = set(actual_counter.keys()) | set(expected_counter.keys())
        intersection_size = sum(
            min(actual_counter[elem], expected_counter[elem]) for elem in all_elements
        )
        union_size = sum(max(actual_counter[elem], expected_counter[elem]) for elem in all_elements)
        jaccard = intersection_size / union_size if union_size else 1.0

        # Check for missing/extra elements (unique elements)
        missing = set(expected_counter.keys()) - set(actual_counter.keys())
        extra = set(actual_counter.keys()) - set(expected_counter.keys())

        feedback_parts = []
        if missing:
            feedback_parts.append(f"missing {len(missing)} element(s)")
        if extra:
            feedback_parts.append(f"{len(extra)} extra element(s)")
        if len(actual) != len(expected):
            feedback_parts.append(f"length {len(actual)} vs expected {len(expected)}")

        feedback = (
            "List mismatch: " + ", ".join(feedback_parts) if feedback_parts else "List mismatch"
        )

        return jaccard, ErrorType.MISSING_EXTRA, feedback

    def _analyze_dict(
        self, actual: dict[str, Any], expected: dict[str, Any]
    ) -> tuple[float, ErrorType, str]:
        """
        Analyze dict outputs using key and value matching.

        Score is calculated as (key_score + value_score) / 2 where:
        - key_score: Jaccard similarity of key sets
        - value_score: Proportion of matching keys with matching values

        Args:
            actual: The actual dict output.
            expected: The expected dict output.

        Returns:
            Tuple of (score, error_type, feedback).
        """
        if actual == expected:
            return 1.0, ErrorType.NONE, "Perfect match"

        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())

        # Calculate key score (Jaccard of keys)
        key_intersection = actual_keys & expected_keys
        key_union = actual_keys | expected_keys

        key_score = len(key_intersection) / len(key_union) if key_union else 1.0

        # Calculate value score (for matching keys, how many values match)
        if key_intersection:
            matching_values = sum(1 for k in key_intersection if actual[k] == expected[k])
            value_score = matching_values / len(key_intersection)
        else:
            value_score = 0.0

        # Combined score
        score = (key_score + value_score) / 2

        # Determine error details
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        wrong_values = [k for k in key_intersection if actual[k] != expected[k]]

        feedback_parts = []
        if missing_keys:
            feedback_parts.append(f"missing keys: {sorted(missing_keys)}")
        if extra_keys:
            feedback_parts.append(f"extra keys: {sorted(extra_keys)}")
        if wrong_values:
            feedback_parts.append(f"wrong values for keys: {sorted(wrong_values)}")

        feedback = (
            "Dict mismatch: " + ", ".join(feedback_parts) if feedback_parts else "Dict mismatch"
        )

        # Classify error type
        if missing_keys or extra_keys or wrong_values:
            error_type = ErrorType.MISSING_EXTRA
        else:
            error_type = ErrorType.NONE

        return score, error_type, feedback

    def _primary_error(self, results: list[ExampleResult]) -> ErrorType:
        """
        Select the highest priority error type from results.

        Priority order (highest to lowest): SYNTAX > SHAPE > MISSING_EXTRA > ORDER > NONE

        Args:
            results: List of example results.

        Returns:
            The highest priority error type found, or NONE if results is empty.
        """
        if not results:
            return ErrorType.NONE

        return max(
            (r.error_type for r in results),
            key=lambda e: self._ERROR_PRIORITY[e],
        )
