"""
Integration tests for algorithmic reviewer diagnosis and scoring.

This module tests the AlgorithmicReviewer class for proper error classification,
score calculation, and feedback generation across various scenarios including
perfect matches, syntax errors, shape mismatches, and partial matches.
"""

from collections.abc import Callable
from typing import Any

from src.domain import ErrorType, Example, Task
from src.reviewer import AlgorithmicReviewer


class TestPerfectMatch:
    """Tests for scenarios where filter output exactly matches expected output."""

    def test_perfect_match_simple_value(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Exact scalar match gets score 1.0 and NONE error type."""
        task = make_task({"x": 42}, 42)
        attempt = reviewer.evaluate(task, ".x")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE
        assert attempt.is_perfect is True

    def test_perfect_match_string(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Exact string match gets score 1.0."""
        task = make_task({"name": "Alice"}, "Alice")
        attempt = reviewer.evaluate(task, ".name")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE
        assert attempt.is_perfect is True

    def test_perfect_match_nested_object(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Exact nested object match gets score 1.0."""
        task = make_task({"user": {"name": "Bob", "age": 30}}, {"name": "Bob", "age": 30})
        attempt = reviewer.evaluate(task, ".user")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE

    def test_perfect_match_array(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Exact array match gets score 1.0."""
        task = make_task({"items": [1, 2, 3]}, [1, 2, 3])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE

    def test_perfect_match_null(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Null output matching null expected gets score 1.0."""
        task = make_task({"value": None}, None)
        attempt = reviewer.evaluate(task, ".value")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE

    def test_perfect_match_boolean(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Boolean output matching expected gets score 1.0."""
        task = make_task({"active": True}, True)
        attempt = reviewer.evaluate(task, ".active")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE


class TestSyntaxError:
    """Tests for scenarios where the jq filter has syntax errors."""

    def test_syntax_error_invalid_brackets(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Invalid jq syntax gets score 0.0 and SYNTAX error type."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, "invalid[[[")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SYNTAX
        assert attempt.is_perfect is False

    def test_syntax_error_unclosed_string(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Unclosed string in filter returns SYNTAX error."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, '."unclosed')

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SYNTAX

    def test_syntax_error_unknown_function(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Unknown function call returns SYNTAX error."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, "nonexistent_func")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SYNTAX

    def test_syntax_error_feedback_contains_error_message(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Syntax error feedback includes jq error message."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, "invalid[[[")

        assert len(attempt.example_results) == 1
        result = attempt.example_results[0]
        assert "jq error" in result.feedback.lower() or "error" in result.feedback.lower()


class TestShapeMismatch:
    """Tests for scenarios where output type doesn't match expected type."""

    def test_shape_mismatch_list_vs_dict(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Type mismatch (list vs dict) classified as SHAPE error."""
        # Expect dict but filter returns list
        task = make_task([1, 2, 3], {"a": 1})
        attempt = reviewer.evaluate(task, ".")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SHAPE

    def test_shape_mismatch_dict_vs_list(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict output when list expected is SHAPE error."""
        task = make_task({"items": {"a": 1}}, [1, 2, 3])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SHAPE

    def test_shape_mismatch_scalar_vs_list(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Scalar output when list expected is SHAPE error."""
        task = make_task({"x": 42}, [42])
        attempt = reviewer.evaluate(task, ".x")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SHAPE

    def test_shape_mismatch_list_vs_scalar(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """List output when scalar expected is SHAPE error."""
        task = make_task({"items": [1, 2]}, 1)
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.SHAPE

    def test_shape_mismatch_feedback_descriptive(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Shape mismatch feedback describes the type difference."""
        task = make_task([1, 2, 3], {"key": "value"})
        attempt = reviewer.evaluate(task, ".")

        result = attempt.example_results[0]
        assert "list" in result.feedback.lower() or "dict" in result.feedback.lower()


class TestMissingKeys:
    """Tests for dict outputs with missing keys."""

    def test_missing_single_key(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict with one missing key gets partial score and MISSING_EXTRA error."""
        task = make_task({"a": 1}, {"a": 1, "b": 2})
        attempt = reviewer.evaluate(task, ".")

        assert 0.0 < attempt.aggregated_score < 1.0
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_missing_multiple_keys(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict with multiple missing keys gets lower partial score."""
        task = make_task({"a": 1}, {"a": 1, "b": 2, "c": 3, "d": 4})
        attempt = reviewer.evaluate(task, ".")

        # Score formula: (key_score + value_score) / 2
        # key_score = 1/4 = 0.25 (Jaccard of keys)
        # value_score = 1/1 = 1.0 (matching values for intersection keys)
        # combined = (0.25 + 1.0) / 2 = 0.625
        assert 0.6 < attempt.aggregated_score < 0.7
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_missing_keys_feedback(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Missing keys are mentioned in feedback."""
        task = make_task({"a": 1}, {"a": 1, "b": 2})
        attempt = reviewer.evaluate(task, ".")

        result = attempt.example_results[0]
        assert "missing" in result.feedback.lower()


class TestExtraKeys:
    """Tests for dict outputs with extra keys."""

    def test_extra_single_key(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict with one extra key gets partial score and MISSING_EXTRA error."""
        task = make_task({"a": 1, "b": 2}, {"a": 1})
        attempt = reviewer.evaluate(task, ".")

        assert 0.0 < attempt.aggregated_score < 1.0
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_extra_multiple_keys(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict with multiple extra keys gets lower partial score."""
        task = make_task({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1})
        attempt = reviewer.evaluate(task, ".")

        # Score formula: (key_score + value_score) / 2
        # key_score = 1/4 = 0.25 (Jaccard of keys)
        # value_score = 1/1 = 1.0 (matching values for intersection keys)
        # combined = (0.25 + 1.0) / 2 = 0.625
        assert 0.6 < attempt.aggregated_score < 0.7
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_extra_keys_feedback(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Extra keys are mentioned in feedback."""
        task = make_task({"a": 1, "extra": 99}, {"a": 1})
        attempt = reviewer.evaluate(task, ".")

        result = attempt.example_results[0]
        assert "extra" in result.feedback.lower()


class TestDictWrongValues:
    """Tests for dict outputs with correct keys but wrong values."""

    def test_wrong_value_single_key(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Dict with correct keys but wrong values gets partial score."""
        task = make_task({"a": 1}, {"a": 2})
        attempt = reviewer.evaluate(task, ".")

        # Key score = 1.0 (both have 'a'), value score = 0.0 (different values)
        # Combined = (1.0 + 0.0) / 2 = 0.5
        assert attempt.aggregated_score == 0.5
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_wrong_values_feedback(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Wrong values are mentioned in feedback."""
        task = make_task({"a": 1, "b": 2}, {"a": 1, "b": 99})
        attempt = reviewer.evaluate(task, ".")

        result = attempt.example_results[0]
        assert "wrong" in result.feedback.lower() or "value" in result.feedback.lower()


class TestOrderMismatch:
    """Tests for list outputs with correct elements but wrong order."""

    def test_order_mismatch_scores_point_eight(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Same elements in wrong order gets score 0.8 and ORDER error."""
        task = make_task({"items": [3, 2, 1]}, [1, 2, 3])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.8
        assert attempt.primary_error == ErrorType.ORDER

    def test_order_mismatch_reversed_list(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Reversed list gets ORDER error with 0.8 score."""
        task = make_task({"data": [5, 4, 3, 2, 1]}, [1, 2, 3, 4, 5])
        attempt = reviewer.evaluate(task, ".data")

        assert attempt.aggregated_score == 0.8
        assert attempt.primary_error == ErrorType.ORDER

    def test_order_mismatch_feedback(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Order mismatch feedback mentions order issue."""
        task = make_task({"items": [2, 1]}, [1, 2])
        attempt = reviewer.evaluate(task, ".items")

        result = attempt.example_results[0]
        assert "order" in result.feedback.lower()

    def test_order_mismatch_with_objects(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Order mismatch with object elements gets 0.8 score."""
        task = make_task(
            {"items": [{"id": 2}, {"id": 1}]},
            [{"id": 1}, {"id": 2}],
        )
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.8
        assert attempt.primary_error == ErrorType.ORDER


class TestPartialListMatch:
    """Tests for partial list matches using Jaccard similarity."""

    def test_jaccard_half_overlap(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Half overlapping elements get Jaccard score around 0.33."""
        # Actual: [1, 2], Expected: [2, 3]
        # Intersection: {2}, Union: {1, 2, 3}
        # Jaccard = 1/3 ≈ 0.333
        task = make_task({"items": [1, 2]}, [2, 3])
        attempt = reviewer.evaluate(task, ".items")

        assert 0.3 <= attempt.aggregated_score <= 0.4
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_jaccard_two_thirds_overlap(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Two-thirds overlapping elements get appropriate Jaccard score."""
        # Actual: [1, 2, 3], Expected: [1, 2, 4]
        # Intersection: {1, 2}, Union: {1, 2, 3, 4}
        # Jaccard = 2/4 = 0.5
        task = make_task({"items": [1, 2, 3]}, [1, 2, 4])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.5
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_jaccard_no_overlap(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """No overlapping elements get Jaccard score 0."""
        task = make_task({"items": [1, 2, 3]}, [4, 5, 6])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_partial_list_feedback_mentions_missing(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Partial list match feedback mentions missing elements."""
        task = make_task({"items": [1, 2]}, [1, 2, 3, 4])
        attempt = reviewer.evaluate(task, ".items")

        result = attempt.example_results[0]
        assert "missing" in result.feedback.lower()

    def test_partial_list_feedback_mentions_extra(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Partial list match feedback mentions extra elements."""
        task = make_task({"items": [1, 2, 3, 4]}, [1, 2])
        attempt = reviewer.evaluate(task, ".items")

        result = attempt.example_results[0]
        assert "extra" in result.feedback.lower()


class TestEmptyCollections:
    """Tests for empty list and dict handling."""

    def test_empty_list_match(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Empty expected and actual lists match perfectly."""
        task = make_task({"items": []}, [])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE
        assert attempt.is_perfect is True

    def test_empty_dict_match(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Empty expected and actual dicts match perfectly."""
        task = make_task({"data": {}}, {})
        attempt = reviewer.evaluate(task, ".data")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE

    def test_empty_vs_nonempty_list(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Empty list vs non-empty list gets MISSING_EXTRA error."""
        task = make_task({"items": []}, [1, 2, 3])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_nonempty_vs_empty_list(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Non-empty list vs empty expected gets MISSING_EXTRA error."""
        task = make_task({"items": [1, 2, 3]}, [])
        attempt = reviewer.evaluate(task, ".items")

        assert attempt.aggregated_score == 0.0
        assert attempt.primary_error == ErrorType.MISSING_EXTRA


class TestMultiExampleAggregation:
    """Tests for score aggregation across multiple examples."""

    def test_multi_example_averaging(self, reviewer: AlgorithmicReviewer):
        """Multiple examples have scores averaged."""
        # Create task with 3 examples where filter matches 2 perfectly
        examples = [
            Example(input_data={"x": 1}, expected_output=1),  # Will match
            Example(input_data={"x": 2}, expected_output=2),  # Will match
            Example(input_data={"x": 3}, expected_output=999),  # Won't match
        ]
        task = Task(id="multi-example", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".x")

        # 2 perfect matches (1.0 each) + 1 mismatch (0.0) = average 0.666...
        assert 0.6 <= attempt.aggregated_score <= 0.7
        assert len(attempt.example_results) == 3

    def test_all_examples_perfect(self, reviewer: AlgorithmicReviewer):
        """All examples matching gives score 1.0."""
        examples = [
            Example(input_data={"x": 1}, expected_output=1),
            Example(input_data={"x": 2}, expected_output=2),
            Example(input_data={"x": 3}, expected_output=3),
        ]
        task = Task(id="all-perfect", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".x")

        assert attempt.aggregated_score == 1.0
        assert attempt.is_perfect is True

    def test_no_examples_match(self, reviewer: AlgorithmicReviewer):
        """No examples matching gives score 0.0."""
        examples = [
            Example(input_data={"x": 1}, expected_output=100),
            Example(input_data={"x": 2}, expected_output=200),
            Example(input_data={"x": 3}, expected_output=300),
        ]
        task = Task(id="none-match", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".x")

        assert attempt.aggregated_score == 0.0
        assert attempt.is_perfect is False

    def test_partial_scores_averaged(self, reviewer: AlgorithmicReviewer):
        """Partial scores are properly averaged."""
        # First example: order mismatch (0.8)
        # Second example: perfect match (1.0)
        examples = [
            Example(input_data={"items": [2, 1]}, expected_output=[1, 2]),
            Example(input_data={"items": [3, 4]}, expected_output=[3, 4]),
        ]
        task = Task(id="partial-scores", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".items")

        # (0.8 + 1.0) / 2 = 0.9
        assert attempt.aggregated_score == 0.9


class TestPrimaryErrorSelection:
    """Tests for primary error type selection based on priority."""

    def test_syntax_error_takes_priority(self, reviewer: AlgorithmicReviewer):
        """SYNTAX error has highest priority."""
        # Create a filter that will fail with syntax error
        examples = [
            Example(input_data={"x": 1}, expected_output=1),
            Example(input_data={"x": 2}, expected_output=2),
        ]
        task = Task(id="syntax-priority", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, "invalid[[[")

        assert attempt.primary_error == ErrorType.SYNTAX

    def test_shape_error_over_missing_extra(self, reviewer: AlgorithmicReviewer):
        """SHAPE error takes priority over MISSING_EXTRA."""
        # First example: shape mismatch (list vs dict)
        # Second example: partial match
        examples = [
            Example(input_data={"items": [1, 2]}, expected_output={"a": 1}),
            Example(input_data={"items": [1, 2]}, expected_output=[1, 2, 3]),
        ]
        task = Task(id="shape-priority", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".items")

        assert attempt.primary_error == ErrorType.SHAPE

    def test_missing_extra_over_order(self, reviewer: AlgorithmicReviewer):
        """MISSING_EXTRA error takes priority over ORDER."""
        examples = [
            Example(input_data={"items": [1, 2]}, expected_output=[2, 1]),  # ORDER
            Example(input_data={"items": [1, 2]}, expected_output=[1, 2, 3]),  # MISSING_EXTRA
        ]
        task = Task(id="missing-priority", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".items")

        assert attempt.primary_error == ErrorType.MISSING_EXTRA

    def test_none_when_all_perfect(self, reviewer: AlgorithmicReviewer):
        """NONE error when all examples are perfect."""
        examples = [
            Example(input_data={"x": 1}, expected_output=1),
            Example(input_data={"x": 2}, expected_output=2),
        ]
        task = Task(id="all-none", description="Test task", examples=examples)

        attempt = reviewer.evaluate(task, ".x")

        assert attempt.primary_error == ErrorType.NONE


class TestExampleResultDetails:
    """Tests for ExampleResult fields being properly populated."""

    def test_actual_output_captured(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """ExampleResult contains actual output from filter."""
        task = make_task({"x": 42}, 42)
        attempt = reviewer.evaluate(task, ".x")

        result = attempt.example_results[0]
        assert result.actual_output == 42

    def test_expected_output_captured(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """ExampleResult contains expected output from task."""
        task = make_task({"x": 42}, 99)
        attempt = reviewer.evaluate(task, ".x")

        result = attempt.example_results[0]
        assert result.expected_output == 99

    def test_feedback_non_empty_on_mismatch(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """ExampleResult feedback is non-empty when there's an error."""
        task = make_task({"x": 1}, 999)
        attempt = reviewer.evaluate(task, ".x")

        result = attempt.example_results[0]
        assert len(result.feedback) > 0

    def test_score_in_valid_range(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """ExampleResult score is always between 0.0 and 1.0."""
        task = make_task({"items": [1, 2, 3]}, [2, 3, 4])
        attempt = reviewer.evaluate(task, ".items")

        result = attempt.example_results[0]
        assert 0.0 <= result.score <= 1.0


class TestAttemptProperties:
    """Tests for Attempt dataclass properties."""

    def test_iteration_set_to_zero(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Reviewer returns Attempt with iteration=0 (orchestrator sets it)."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, ".x")

        assert attempt.iteration == 0

    def test_filter_code_stored(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Attempt stores the filter code that was evaluated."""
        task = make_task({"x": 1}, 1)
        attempt = reviewer.evaluate(task, ".x")

        assert attempt.filter_code == ".x"

    def test_example_results_count_matches_task(
        self,
        reviewer: AlgorithmicReviewer,
    ):
        """Number of ExampleResults matches number of examples in task."""
        examples = [Example(input_data={"x": i}, expected_output=i) for i in range(5)]
        task = Task(id="five-examples", description="Test", examples=examples)

        attempt = reviewer.evaluate(task, ".x")

        assert len(attempt.example_results) == 5


class TestSpecialCases:
    """Tests for edge cases and special scenarios."""

    def test_multiline_jq_output(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Filter producing multi-line output is handled correctly."""
        # .[].x produces one value per line
        task = make_task([{"x": 1}, {"x": 2}, {"x": 3}], [1, 2, 3])
        attempt = reviewer.evaluate(task, "[.[].x]")

        assert attempt.aggregated_score == 1.0

    def test_filter_returning_null_for_missing_field(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Accessing missing field returns null which may not match expected."""
        task = make_task({}, "expected_value")
        attempt = reviewer.evaluate(task, ".missing")

        # null != "expected_value"
        assert attempt.aggregated_score == 0.0

    def test_complex_nested_comparison(
        self,
        reviewer: AlgorithmicReviewer,
        make_task: Callable[[Any, Any, str], Task],
    ):
        """Complex nested structures are compared correctly."""
        expected = {
            "users": [{"name": "Alice", "roles": ["admin", "user"]}],
            "count": 1,
        }
        task = make_task(expected, expected)
        attempt = reviewer.evaluate(task, ".")

        assert attempt.aggregated_score == 1.0
        assert attempt.primary_error == ErrorType.NONE
