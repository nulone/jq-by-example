"""
Edge case tests for production-ready jq-by-example.

This module contains comprehensive edge case tests covering:
- Null input/output handling
- Empty arrays and objects
- Deeply nested structures (3+ levels)
- Special characters in keys
- Large arrays (100+ items)
- Type mismatches
"""

from unittest.mock import MagicMock

import pytest

from src.domain import Example, Task
from src.executor import JQExecutor
from src.generator import JQGenerator
from src.orchestrator import Orchestrator
from src.reviewer import AlgorithmicReviewer


@pytest.fixture
def orchestrator_with_mock_generator(executor: JQExecutor) -> tuple[Orchestrator, MagicMock]:
    """
    Create orchestrator with mocked generator for edge case testing.

    Returns:
        Tuple of (orchestrator, mock_generator) for test control.
    """
    mock_generator = MagicMock(spec=JQGenerator)
    reviewer = AlgorithmicReviewer(executor)
    orchestrator = Orchestrator(
        generator=mock_generator,
        reviewer=reviewer,
        max_iterations=10,
        stagnation_limit=3,
    )
    return orchestrator, mock_generator


class TestNullHandling:
    """Tests for null input and output handling."""

    def test_null_input(self, orchestrator_with_mock_generator):
        """Filter should handle null input gracefully."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="null-input",
            description="Return null input as-is",
            examples=[
                Example(input_data=None, expected_output=None),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True
        assert solution.best_filter == "."

    def test_null_output(self, orchestrator_with_mock_generator):
        """Filter should produce null output when expected."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = ".nonexistent"

        task = Task(
            id="null-output",
            description="Extract nonexistent field (returns null)",
            examples=[
                Example(input_data={"foo": "bar"}, expected_output=None),
                Example(input_data={"baz": 123}, expected_output=None),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_null_in_array(self, orchestrator_with_mock_generator):
        """Filter should handle arrays containing null values."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="null-in-array",
            description="Return array with nulls as-is",
            examples=[
                Example(
                    input_data=[1, None, 3, None, 5],
                    expected_output=[1, None, 3, None, 5],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_null_object_values(self, orchestrator_with_mock_generator):
        """Filter should handle objects with null values."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="null-object-values",
            description="Return object with null values as-is",
            examples=[
                Example(
                    input_data={"name": "Alice", "email": None, "age": 30},
                    expected_output={"name": "Alice", "email": None, "age": 30},
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True


class TestEmptyCollections:
    """Tests for empty arrays and objects."""

    def test_empty_array_input(self, orchestrator_with_mock_generator):
        """Filter should handle empty array input."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="empty-array-input",
            description="Return empty array as-is",
            examples=[
                Example(input_data=[], expected_output=[]),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_empty_array_output(self, orchestrator_with_mock_generator):
        """Filter should produce empty array output when needed."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "[.items[] | select(.active == true)]"

        task = Task(
            id="empty-array-output",
            description="Filter items that are active (none match)",
            examples=[
                Example(
                    input_data={"items": [{"id": 1, "active": False}]},
                    expected_output=[],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_empty_object_input(self, orchestrator_with_mock_generator):
        """Filter should handle empty object input."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="empty-object-input",
            description="Return empty object as-is",
            examples=[
                Example(input_data={}, expected_output={}),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_empty_object_output(self, orchestrator_with_mock_generator):
        """Filter should produce empty object when needed."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "{}"

        task = Task(
            id="empty-object-output",
            description="Create empty object",
            examples=[
                Example(input_data={"foo": "bar"}, expected_output={}),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_nested_empty_collections(self, orchestrator_with_mock_generator):
        """Filter should handle nested empty arrays and objects."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "."

        task = Task(
            id="nested-empty",
            description="Return nested empty collections as-is",
            examples=[
                Example(
                    input_data={"items": [], "meta": {}},
                    expected_output={"items": [], "meta": {}},
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True


class TestDeeplyNested:
    """Tests for deeply nested structures (3+ levels)."""

    def test_extract_from_3_levels(self, orchestrator_with_mock_generator):
        """Filter should extract value from 3-level nesting."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = ".user.profile.name"

        task = Task(
            id="3-level-extract",
            description="Extract name from 3-level nested structure",
            examples=[
                Example(
                    input_data={"user": {"profile": {"name": "Alice", "age": 30}}},
                    expected_output="Alice",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_extract_from_5_levels(self, orchestrator_with_mock_generator):
        """Filter should extract value from 5-level nesting."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = ".data.user.account.settings.theme"

        task = Task(
            id="5-level-extract",
            description="Extract theme from deeply nested structure",
            examples=[
                Example(
                    input_data={
                        "data": {"user": {"account": {"settings": {"theme": "dark", "lang": "en"}}}}
                    },
                    expected_output="dark",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_nested_array_of_objects(self, orchestrator_with_mock_generator):
        """Filter should handle arrays nested in objects 3+ levels deep."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = ".company.departments[].employees[].name"

        task = Task(
            id="nested-array-objects",
            description="Extract all employee names from nested structure",
            examples=[
                Example(
                    input_data={
                        "company": {
                            "departments": [
                                {
                                    "name": "Engineering",
                                    "employees": [
                                        {"name": "Alice"},
                                        {"name": "Bob"},
                                    ],
                                }
                            ]
                        }
                    },
                    expected_output=["Alice", "Bob"],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_deeply_nested_with_nulls(self, orchestrator_with_mock_generator):
        """Filter should handle null values in deeply nested paths."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        # Using // to provide default when path is null
        mock_gen.generate.return_value = '.a.b.c.d // "default"'

        task = Task(
            id="deep-nest-null",
            description="Extract deeply nested value with null handling",
            examples=[
                Example(
                    input_data={"a": {"b": None}},
                    expected_output="default",
                ),
                Example(
                    input_data={"a": {"b": {"c": {"d": "found"}}}},
                    expected_output="found",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True


class TestSpecialCharactersInKeys:
    """Tests for special characters in object keys."""

    def test_keys_with_spaces(self, orchestrator_with_mock_generator):
        """Filter should handle keys with spaces."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '.["user name"]'

        task = Task(
            id="keys-with-spaces",
            description="Extract field with spaces in key name",
            examples=[
                Example(
                    input_data={"user name": "Alice", "age": 30},
                    expected_output="Alice",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_keys_with_special_chars(self, orchestrator_with_mock_generator):
        """Filter should handle keys with special characters."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '.["email@address"]'

        task = Task(
            id="keys-special-chars",
            description="Extract field with special characters in key",
            examples=[
                Example(
                    input_data={
                        "email@address": "alice@example.com",
                        "name": "Alice",
                    },
                    expected_output="alice@example.com",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_keys_with_unicode(self, orchestrator_with_mock_generator):
        """Filter should handle keys with unicode characters."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '.["名前"]'

        task = Task(
            id="keys-unicode",
            description="Extract field with unicode key name",
            examples=[
                Example(
                    input_data={"名前": "太郎", "age": 25},
                    expected_output="太郎",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_keys_with_hyphens_and_dots(self, orchestrator_with_mock_generator):
        """Filter should handle keys with hyphens and dots."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '.["content-type"]'

        task = Task(
            id="keys-hyphens-dots",
            description="Extract field with hyphens in key",
            examples=[
                Example(
                    input_data={"content-type": "application/json"},
                    expected_output="application/json",
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True


class TestLargeArrays:
    """Tests for large arrays with 100+ items."""

    def test_large_array_filter(self, orchestrator_with_mock_generator):
        """Filter should handle arrays with 100+ items."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "[.[] | select(. % 2 == 0)]"

        # Create array with 150 items
        large_input = list(range(1, 151))
        expected_output = [x for x in large_input if x % 2 == 0]

        task = Task(
            id="large-array-filter",
            description="Filter even numbers from large array",
            examples=[
                Example(input_data=large_input, expected_output=expected_output),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_large_array_map(self, orchestrator_with_mock_generator):
        """Filter should transform large arrays efficiently."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "[.[] | . * 2]"

        # Create array with 200 items
        large_input = list(range(1, 201))
        expected_output = [x * 2 for x in large_input]

        task = Task(
            id="large-array-map",
            description="Double all numbers in large array",
            examples=[
                Example(input_data=large_input, expected_output=expected_output),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_large_array_of_objects(self, orchestrator_with_mock_generator):
        """Filter should handle large arrays of objects."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "[.[].id]"

        # Create array with 100 objects
        large_input = [{"id": i, "value": f"item_{i}"} for i in range(100)]
        expected_output = list(range(100))

        task = Task(
            id="large-array-objects",
            description="Extract IDs from large array of objects",
            examples=[
                Example(input_data=large_input, expected_output=expected_output),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True


class TestTypeMismatches:
    """Tests for type mismatches and conversions."""

    def test_string_to_number_comparison(self, orchestrator_with_mock_generator):
        """Filter should handle string vs number comparisons."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        # This filter will work because jq handles type coercion
        mock_gen.generate.return_value = "[.[] | select(.age > 25)]"

        task = Task(
            id="type-comparison",
            description="Filter objects where age > 25",
            examples=[
                Example(
                    input_data=[
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 20},
                        {"name": "Charlie", "age": 35},
                    ],
                    expected_output=[
                        {"name": "Alice", "age": 30},
                        {"name": "Charlie", "age": 35},
                    ],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_mixed_types_in_array(self, orchestrator_with_mock_generator):
        """Filter should handle arrays with mixed types."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '[.[] | select(type == "number")]'

        task = Task(
            id="mixed-types-filter",
            description="Extract only numbers from mixed-type array",
            examples=[
                Example(
                    input_data=[1, "hello", 2, None, 3, True, 4],
                    expected_output=[1, 2, 3, 4],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_boolean_vs_number(self, orchestrator_with_mock_generator):
        """Filter should distinguish between boolean and number types."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = '[.[] | select(type == "boolean")]'

        task = Task(
            id="boolean-filter",
            description="Extract only boolean values",
            examples=[
                Example(
                    input_data=[True, 1, False, 0, True],
                    expected_output=[True, False, True],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_string_number_mixed_output(self, orchestrator_with_mock_generator):
        """Filter should handle conversion between string and number."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = "[.[] | tonumber]"

        task = Task(
            id="string-to-number",
            description="Convert string numbers to actual numbers",
            examples=[
                Example(
                    input_data=["1", "2", "3"],
                    expected_output=[1, 2, 3],
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True

    def test_array_vs_single_value(self, orchestrator_with_mock_generator):
        """Filter should handle returning single value vs array properly."""
        orchestrator, mock_gen = orchestrator_with_mock_generator
        mock_gen.generate.return_value = ".[0]"

        task = Task(
            id="array-to-value",
            description="Extract first element from array",
            examples=[
                Example(
                    input_data=[10, 20, 30],
                    expected_output=10,
                ),
            ],
        )

        solution = orchestrator.solve(task)
        assert solution.success is True
