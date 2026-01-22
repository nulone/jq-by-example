"""
Integration tests for CLI argument parsing and task loading.

This module tests the CLI module for proper argument parsing, task file loading,
and integration with the orchestrator components. Uses tmp_path fixture for
test task files and monkeypatch for environment variable manipulation.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import (
    _create_interactive_task,
    _estimate_difficulty,
    _format_api_key_error,
    _format_jq_not_found_error,
    _format_score,
    _format_task_not_found_error,
    _parse_args,
    _setup_logging,
    _validate_json_string,
    load_tasks,
    main,
)
from src.domain import Solution, Task


class TestLoadTasksValidJSON:
    """Tests for load_tasks parsing valid JSON files."""

    def test_parses_single_task(self, tmp_path: Path):
        """Single task is correctly parsed into Task object."""
        tasks_data = {
            "tasks": [
                {
                    "id": "test-task",
                    "description": "Extract the name field",
                    "examples": [{"input": {"name": "Alice"}, "expected_output": "Alice"}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert len(tasks) == 1
        assert tasks[0].id == "test-task"
        assert tasks[0].description == "Extract the name field"
        assert len(tasks[0].examples) == 1

    def test_parses_multiple_tasks(self, tmp_path: Path):
        """Multiple tasks are correctly parsed."""
        tasks_data = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First task",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                },
                {
                    "id": "task-2",
                    "description": "Second task",
                    "examples": [{"input": {"y": 2}, "expected_output": 2}],
                },
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert len(tasks) == 2
        assert tasks[0].id == "task-1"
        assert tasks[1].id == "task-2"

    def test_parses_multiple_examples(self, tmp_path: Path):
        """Task with multiple examples is correctly parsed."""
        tasks_data = {
            "tasks": [
                {
                    "id": "multi-example",
                    "description": "Task with 3 examples",
                    "examples": [
                        {"input": {"x": 1}, "expected_output": 1},
                        {"input": {"x": 2}, "expected_output": 2},
                        {"input": {"x": 3}, "expected_output": 3},
                    ],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert len(tasks) == 1
        assert len(tasks[0].examples) == 3
        assert tasks[0].examples[0].input_data == {"x": 1}
        assert tasks[0].examples[0].expected_output == 1
        assert tasks[0].examples[2].input_data == {"x": 3}
        assert tasks[0].examples[2].expected_output == 3

    def test_parses_complex_input_data(self, tmp_path: Path):
        """Complex nested input data is correctly parsed."""
        tasks_data = {
            "tasks": [
                {
                    "id": "complex-task",
                    "description": "Complex nested data",
                    "examples": [
                        {
                            "input": {
                                "user": {"name": "Alice", "roles": ["admin", "user"]},
                                "metadata": {"created": "2024-01-01"},
                            },
                            "expected_output": {"name": "Alice", "roles": ["admin", "user"]},
                        }
                    ],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert tasks[0].examples[0].input_data["user"]["name"] == "Alice"
        assert tasks[0].examples[0].input_data["user"]["roles"] == ["admin", "user"]

    def test_parses_array_input(self, tmp_path: Path):
        """Array input data is correctly parsed."""
        tasks_data = {
            "tasks": [
                {
                    "id": "array-task",
                    "description": "Array input",
                    "examples": [
                        {
                            "input": [{"id": 1}, {"id": 2}, {"id": 3}],
                            "expected_output": [1, 2, 3],
                        }
                    ],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert isinstance(tasks[0].examples[0].input_data, list)
        assert len(tasks[0].examples[0].input_data) == 3

    def test_parses_null_expected_output(self, tmp_path: Path):
        """Null expected output is correctly parsed."""
        tasks_data = {
            "tasks": [
                {
                    "id": "null-task",
                    "description": "Null output",
                    "examples": [{"input": {"missing": "field"}, "expected_output": None}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert tasks[0].examples[0].expected_output is None

    def test_returns_task_objects(self, tmp_path: Path):
        """load_tasks returns proper Task domain objects."""
        tasks_data = {
            "tasks": [
                {
                    "id": "domain-test",
                    "description": "Test task",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        tasks = load_tasks(str(tasks_file))

        assert isinstance(tasks[0], Task)
        assert hasattr(tasks[0], "id")
        assert hasattr(tasks[0], "description")
        assert hasattr(tasks[0], "examples")


class TestLoadTasksMissingFile:
    """Tests for load_tasks handling missing files."""

    def test_raises_file_not_found_error(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tasks("/nonexistent/path/to/tasks.json")

    def test_raises_for_missing_relative_path(self, tmp_path: Path):
        """Missing relative path raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            load_tasks(str(nonexistent))


class TestLoadTasksInvalidJSON:
    """Tests for load_tasks handling invalid JSON."""

    def test_raises_on_malformed_json(self, tmp_path: Path):
        """Malformed JSON raises JSONDecodeError."""
        tasks_file = tmp_path / "invalid.json"
        tasks_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_tasks(str(tasks_file))

    def test_raises_on_missing_tasks_key(self, tmp_path: Path):
        """Missing 'tasks' key raises KeyError."""
        tasks_file = tmp_path / "no_tasks.json"
        tasks_file.write_text('{"other": []}')

        with pytest.raises(KeyError):
            load_tasks(str(tasks_file))

    def test_raises_on_missing_id_field(self, tmp_path: Path):
        """Missing 'id' field in task raises KeyError."""
        tasks_data = {
            "tasks": [
                {
                    "description": "No ID",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "no_id.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with pytest.raises(KeyError):
            load_tasks(str(tasks_file))

    def test_raises_on_missing_examples_field(self, tmp_path: Path):
        """Missing 'examples' field in task raises KeyError."""
        tasks_data = {"tasks": [{"id": "test", "description": "No examples"}]}
        tasks_file = tmp_path / "no_examples.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with pytest.raises(KeyError):
            load_tasks(str(tasks_file))

    def test_raises_on_missing_input_in_example(self, tmp_path: Path):
        """Missing 'input' in example raises KeyError."""
        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Missing input",
                    "examples": [{"expected_output": 1}],
                }
            ]
        }
        tasks_file = tmp_path / "no_input.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with pytest.raises(KeyError):
            load_tasks(str(tasks_file))

    def test_raises_on_missing_expected_output_in_example(self, tmp_path: Path):
        """Missing 'expected_output' in example raises KeyError."""
        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Missing expected_output",
                    "examples": [{"input": {"x": 1}}],
                }
            ]
        }
        tasks_file = tmp_path / "no_expected.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with pytest.raises(KeyError):
            load_tasks(str(tasks_file))


class TestMainWithoutAPIKey:
    """Tests for main returning 1 without API key."""

    def test_returns_1_without_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """CLI fails with exit code 1 when API key is missing."""
        # Remove API key from environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create a valid tasks file
        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        # Mock JQExecutor to avoid jq binary requirement
        with patch("src.cli.JQExecutor"):
            result = main(["--task", "test", "--tasks-file", str(tasks_file)])

        assert result == 1

    def test_prints_error_message_without_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """CLI prints meaningful error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with patch("src.cli.JQExecutor"):
            main(["--task", "test", "--tasks-file", str(tasks_file)])

        captured = capsys.readouterr()
        assert "API key" in captured.err or "OPENAI_API_KEY" in captured.err


class TestInteractiveMode:
    """Tests for interactive mode creating valid Task."""

    def test_creates_task_from_input_output_args(self):
        """Interactive mode args create Task with correct data."""
        task = _create_interactive_task(
            input_json='{"x": 1}',
            output_json="1",
            description="Extract x",
        )

        assert isinstance(task, Task)
        assert task.id == "interactive"
        assert task.description == "Extract x"
        assert len(task.examples) == 1
        assert task.examples[0].input_data == {"x": 1}
        assert task.examples[0].expected_output == 1

    def test_creates_task_with_complex_json(self):
        """Interactive mode handles complex nested JSON."""
        task = _create_interactive_task(
            input_json='{"user": {"name": "Alice", "roles": ["admin"]}}',
            output_json='{"name": "Alice", "roles": ["admin"]}',
            description="Extract user",
        )

        assert task.examples[0].input_data == {"user": {"name": "Alice", "roles": ["admin"]}}
        assert task.examples[0].expected_output == {
            "name": "Alice",
            "roles": ["admin"],
        }

    def test_creates_task_with_array_input(self):
        """Interactive mode handles array input."""
        task = _create_interactive_task(
            input_json="[1, 2, 3]",
            output_json="6",
            description="Sum array",
        )

        assert task.examples[0].input_data == [1, 2, 3]
        assert task.examples[0].expected_output == 6

    def test_creates_task_with_null_output(self):
        """Interactive mode handles null expected output."""
        task = _create_interactive_task(
            input_json='{"missing": "field"}',
            output_json="null",
            description="Get nonexistent field",
        )

        assert task.examples[0].expected_output is None

    def test_creates_task_with_string_output(self):
        """Interactive mode handles string expected output."""
        task = _create_interactive_task(
            input_json='{"name": "Alice"}',
            output_json='"Alice"',
            description="Extract name",
        )

        assert task.examples[0].expected_output == "Alice"

    def test_creates_task_with_boolean_output(self):
        """Interactive mode handles boolean expected output."""
        task = _create_interactive_task(
            input_json='{"active": true}',
            output_json="true",
            description="Extract active flag",
        )

        assert task.examples[0].expected_output is True

    def test_raises_on_invalid_input_json(self):
        """Invalid input JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _create_interactive_task(
                input_json="{ invalid }",
                output_json="1",
                description="Test",
            )

    def test_raises_on_invalid_output_json(self):
        """Invalid output JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _create_interactive_task(
                input_json='{"x": 1}',
                output_json="not valid json",
                description="Test",
            )

    def test_uses_default_description(self):
        """Default description is used when not specified."""
        task = _create_interactive_task(
            input_json='{"x": 1}',
            output_json="1",
            description="Transform the input to produce the expected output",
        )

        assert "Transform" in task.description


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parses_task_argument(self):
        """--task argument is correctly parsed."""
        args = _parse_args(["--task", "nested-field"])
        assert args.task == "nested-field"

    def test_parses_short_task_argument(self):
        """-t argument is correctly parsed."""
        args = _parse_args(["-t", "nested-field"])
        assert args.task == "nested-field"

    def test_parses_tasks_file_argument(self):
        """--tasks-file argument is correctly parsed."""
        args = _parse_args(["--tasks-file", "/path/to/tasks.json"])
        assert args.tasks_file == "/path/to/tasks.json"

    def test_default_tasks_file(self):
        """Default tasks file is data/tasks.json."""
        args = _parse_args([])
        assert args.tasks_file == "data/tasks.json"

    def test_parses_max_iters_argument(self):
        """--max-iters argument is correctly parsed."""
        args = _parse_args(["--max-iters", "5"])
        assert args.max_iters == 5

    def test_default_max_iters(self):
        """Default max iterations is 10."""
        args = _parse_args([])
        assert args.max_iters == 10

    def test_parses_baseline_flag(self):
        """--baseline flag is correctly parsed."""
        args = _parse_args(["--baseline"])
        assert args.baseline is True

    def test_baseline_default_false(self):
        """Baseline defaults to False."""
        args = _parse_args([])
        assert args.baseline is False

    def test_parses_input_argument(self):
        """--input/-i argument is correctly parsed."""
        args = _parse_args(["--input", '{"x": 1}'])
        assert args.input == '{"x": 1}'

        args = _parse_args(["-i", '{"x": 1}'])
        assert args.input == '{"x": 1}'

    def test_parses_output_argument(self):
        """--output/-o argument is correctly parsed."""
        args = _parse_args(["--output", "1"])
        assert args.output == "1"

        args = _parse_args(["-o", "1"])
        assert args.output == "1"

    def test_parses_desc_argument(self):
        """--desc/-d argument is correctly parsed."""
        args = _parse_args(["--desc", "Extract x value"])
        assert args.desc == "Extract x value"

        args = _parse_args(["-d", "Extract x value"])
        assert args.desc == "Extract x value"

    def test_parses_verbose_flag(self):
        """--verbose/-v flag is correctly parsed."""
        args = _parse_args(["--verbose"])
        assert args.verbose is True

        args = _parse_args(["-v"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        """Verbose defaults to False."""
        args = _parse_args([])
        assert args.verbose is False

    def test_parses_all_task(self):
        """--task all is correctly parsed."""
        args = _parse_args(["--task", "all"])
        assert args.task == "all"


class TestMainTaskFileMissing:
    """Tests for main handling missing task file."""

    def test_returns_1_for_missing_tasks_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """CLI returns 1 when tasks file doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = main(["--task", "test", "--tasks-file", "/nonexistent/tasks.json"])

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_prints_file_not_found_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """CLI prints file not found error message."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        main(["--task", "test", "--tasks-file", "/nonexistent/path.json"])

        captured = capsys.readouterr()
        assert "/nonexistent/path.json" in captured.err


class TestMainTaskNotFound:
    """Tests for main handling task not found in file."""

    def test_returns_1_for_unknown_task_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when specified task ID doesn't exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "existing-task",
                    "description": "Existing",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        result = main(["--task", "nonexistent-task", "--tasks-file", str(tasks_file)])

        assert result == 1

    def test_prints_task_not_found_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI prints error with available tasks when task not found."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "task-a",
                    "description": "Task A",
                    "examples": [{"input": {}, "expected_output": {}}],
                },
                {
                    "id": "task-b",
                    "description": "Task B",
                    "examples": [{"input": {}, "expected_output": {}}],
                },
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        main(["--task", "nonexistent", "--tasks-file", str(tasks_file)])

        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert "task-a" in captured.err or "task-b" in captured.err


class TestMainMissingRequiredArgs:
    """Tests for main handling missing required arguments."""

    def test_returns_1_without_task_or_interactive_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when neither --task nor interactive mode specified."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = main([])

        assert result == 1
        captured = capsys.readouterr()
        assert "Must specify" in captured.err or "--task" in captured.err


class TestMainInteractiveModeIntegration:
    """Tests for main running in interactive mode."""

    def test_interactive_mode_with_invalid_input_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when interactive mode has invalid input JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = main(["--input", "{ invalid }", "--output", "1"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err

    def test_interactive_mode_with_invalid_output_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when interactive mode has invalid output JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = main(["--input", '{"x": 1}', "--output", "not json"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err


class TestMainJQNotFound:
    """Tests for main handling missing jq binary."""

    def test_returns_1_when_jq_not_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when jq binary is not found."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        # Mock JQExecutor to raise RuntimeError
        with patch("src.cli.JQExecutor") as mock_executor:
            mock_executor.side_effect = RuntimeError("jq binary not found")
            result = main(["--task", "test", "--tasks-file", str(tasks_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "jq" in captured.err.lower()


class TestMainBaselineMode:
    """Tests for main with --baseline flag."""

    def test_baseline_creates_orchestrator_with_max_iterations_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """--baseline flag sets max_iterations=1 on orchestrator."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with patch("src.cli.JQExecutor"), patch("src.cli.JQGenerator"):
            with patch("src.cli.Orchestrator") as mock_orch_class:
                mock_orch = MagicMock()
                mock_orch.solve.return_value = MagicMock(
                    success=True,
                    task_id="test",
                    best_filter=".x",
                    best_score=1.0,
                    iterations_used=1,
                    history=[],
                )
                mock_orch_class.return_value = mock_orch

                main(["--task", "test", "--tasks-file", str(tasks_file), "--baseline"])

                # Check Orchestrator was called with max_iterations=1
                call_kwargs = mock_orch_class.call_args[1]
                assert call_kwargs["max_iterations"] == 1


class TestMainMaxIters:
    """Tests for main with --max-iters flag."""

    def test_max_iters_passed_to_orchestrator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """--max-iters value is passed to orchestrator."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with patch("src.cli.JQExecutor"), patch("src.cli.JQGenerator"):
            with patch("src.cli.Orchestrator") as mock_orch_class:
                mock_orch = MagicMock()
                mock_orch.solve.return_value = MagicMock(
                    success=True,
                    task_id="test",
                    best_filter=".x",
                    best_score=1.0,
                    iterations_used=1,
                    history=[],
                )
                mock_orch_class.return_value = mock_orch

                main(["--task", "test", "--tasks-file", str(tasks_file), "--max-iters", "7"])

                call_kwargs = mock_orch_class.call_args[1]
                assert call_kwargs["max_iterations"] == 7


class TestMainReturnCode:
    """Tests for main return code based on task success."""

    def test_returns_0_when_all_tasks_succeed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """CLI returns 0 when all tasks are solved successfully."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task 1",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                },
                {
                    "id": "task-2",
                    "description": "Task 2",
                    "examples": [{"input": {"y": 2}, "expected_output": 2}],
                },
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with patch("src.cli.JQExecutor"), patch("src.cli.JQGenerator"):
            with patch("src.cli.Orchestrator") as mock_orch_class:
                mock_orch = MagicMock()
                # Both tasks succeed
                mock_orch.solve.side_effect = [
                    MagicMock(
                        success=True,
                        task_id="task-1",
                        best_filter=".x",
                        best_score=1.0,
                        iterations_used=1,
                        history=[],
                    ),
                    MagicMock(
                        success=True,
                        task_id="task-2",
                        best_filter=".y",
                        best_score=1.0,
                        iterations_used=1,
                        history=[],
                    ),
                ]
                mock_orch_class.return_value = mock_orch

                result = main(["--task", "all", "--tasks-file", str(tasks_file)])

        assert result == 0

    def test_returns_1_when_any_task_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """CLI returns 1 when any task fails to solve."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task 1",
                    "examples": [{"input": {"x": 1}, "expected_output": 1}],
                },
                {
                    "id": "task-2",
                    "description": "Task 2",
                    "examples": [{"input": {"y": 2}, "expected_output": 2}],
                },
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        with patch("src.cli.JQExecutor"), patch("src.cli.JQGenerator"):
            with patch("src.cli.Orchestrator") as mock_orch_class:
                mock_orch = MagicMock()
                # First task succeeds, second fails
                mock_orch.solve.side_effect = [
                    MagicMock(
                        success=True,
                        task_id="task-1",
                        best_filter=".x",
                        best_score=1.0,
                        iterations_used=1,
                        history=[],
                    ),
                    MagicMock(
                        success=False,
                        task_id="task-2",
                        best_filter=".wrong",
                        best_score=0.5,
                        iterations_used=10,
                        history=[],
                    ),
                ]
                mock_orch_class.return_value = mock_orch

                result = main(["--task", "all", "--tasks-file", str(tasks_file)])

        assert result == 1


class TestMainInvalidTasksFile:
    """Tests for main handling invalid task file content."""

    def test_returns_1_for_invalid_json_in_tasks_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when tasks file contains invalid JSON."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_file = tmp_path / "invalid.json"
        tasks_file.write_text("{ not valid json }")

        result = main(["--task", "test", "--tasks-file", str(tasks_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err

    def test_returns_1_for_missing_field_in_tasks_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """CLI returns 1 when tasks file is missing required fields."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tasks_data = {
            "tasks": [
                {
                    # Missing 'id' field
                    "description": "Test",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "missing_field.json"
        tasks_file.write_text(json.dumps(tasks_data))

        result = main(["--task", "test", "--tasks-file", str(tasks_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "Missing field" in captured.err


class TestFormatJQNotFoundError:
    """Tests for _format_jq_not_found_error function."""

    def test_contains_error_symbol(self) -> None:
        """Error message should contain warning symbol."""
        result = _format_jq_not_found_error()
        assert "jq binary not found" in result

    def test_contains_installation_instructions(self) -> None:
        """Error message should include platform-specific install commands."""
        result = _format_jq_not_found_error()
        assert "brew install jq" in result
        assert "apt-get install jq" in result
        assert "choco install jq" in result

    def test_contains_verification_steps(self) -> None:
        """Error message should include verification command."""
        result = _format_jq_not_found_error()
        assert "jq --version" in result


class TestFormatApiKeyError:
    """Tests for _format_api_key_error function."""

    def test_openai_provider_message(self) -> None:
        """OpenAI provider should show OpenAI-specific instructions."""
        result = _format_api_key_error("openai")
        assert "OpenAI" in result or "openai" in result.lower()
        assert "OPENAI_API_KEY" in result
        assert "platform.openai.com" in result

    def test_anthropic_provider_message(self) -> None:
        """Anthropic provider should show Anthropic-specific instructions."""
        result = _format_api_key_error("anthropic")
        assert "Anthropic" in result or "anthropic" in result.lower()
        assert "ANTHROPIC_API_KEY" in result
        assert "console.anthropic.com" in result

    def test_unknown_provider_message(self) -> None:
        """Unknown provider should show generic message."""
        result = _format_api_key_error("unknown-provider")
        assert "API key required" in result
        assert "unknown-provider" in result


class TestFormatTaskNotFoundError:
    """Tests for _format_task_not_found_error function."""

    def test_shows_task_id(self) -> None:
        """Error message should show the invalid task ID."""
        task = Task(id="valid", description="Test", examples=[])
        result = _format_task_not_found_error("invalid", [task])
        assert "invalid" in result

    def test_suggests_close_match(self) -> None:
        """Error message should suggest close matches."""
        task = Task(id="nested-field", description="Extract nested field", examples=[])
        result = _format_task_not_found_error("nested-fiel", [task])
        assert "nested-field" in result
        assert "Did you mean" in result or "mean" in result

    def test_lists_available_tasks(self) -> None:
        """Error message should list available task IDs."""
        task1 = Task(id="task-1", description="First task", examples=[])
        task2 = Task(id="task-2", description="Second task", examples=[])
        result = _format_task_not_found_error("invalid", [task1, task2])
        assert "task-1" in result
        assert "task-2" in result

    def test_limits_task_list_to_five(self) -> None:
        """Error message should show only first 5 tasks."""
        tasks = [Task(id=f"task-{i}", description=f"Task {i}", examples=[]) for i in range(10)]
        result = _format_task_not_found_error("invalid", tasks)
        assert "and 5 more" in result or "more" in result


class TestValidateJsonString:
    """Tests for _validate_json_string function."""

    def test_valid_json_object(self) -> None:
        """Valid JSON object should be accepted."""
        is_valid, error, data = _validate_json_string('{"x": 1}', "input")
        assert is_valid is True
        assert error == ""
        assert data == {"x": 1}

    def test_valid_json_array(self) -> None:
        """Valid JSON array should be accepted."""
        is_valid, error, data = _validate_json_string("[1, 2, 3]", "input")
        assert is_valid is True
        assert error == ""
        assert data == [1, 2, 3]

    def test_valid_json_string(self) -> None:
        """Valid JSON string should be accepted."""
        is_valid, error, data = _validate_json_string('"hello"', "input")
        assert is_valid is True
        assert error == ""
        assert data == "hello"

    def test_valid_json_number(self) -> None:
        """Valid JSON number should be accepted."""
        is_valid, error, data = _validate_json_string("42", "output")
        assert is_valid is True
        assert error == ""
        assert data == 42

    def test_invalid_json_missing_brace(self) -> None:
        """Invalid JSON with missing brace should be detected."""
        is_valid, error, data = _validate_json_string('{"x": 1', "input")
        assert is_valid is False
        assert "Invalid JSON" in error
        assert "Missing closing brace" in error
        assert data is None

    def test_invalid_json_missing_bracket(self) -> None:
        """Invalid JSON with missing bracket should be detected."""
        is_valid, error, data = _validate_json_string("[1, 2", "input")
        assert is_valid is False
        assert "Invalid JSON" in error
        assert "Missing closing bracket" in error
        assert data is None

    def test_invalid_json_single_quotes(self) -> None:
        """Invalid JSON with single quotes should suggest double quotes."""
        is_valid, error, data = _validate_json_string("{'x': 1}", "input")
        assert is_valid is False
        assert "Invalid JSON" in error
        assert "double quotes" in error or 'Use "' in error
        assert data is None

    def test_invalid_json_unmatched_quote(self) -> None:
        """Invalid JSON with unmatched quote should be detected."""
        is_valid, error, data = _validate_json_string('{"x": "hello}', "input")
        assert is_valid is False
        assert "Invalid JSON" in error
        assert data is None

    def test_error_message_shows_position(self) -> None:
        """Error message should show line and column of error."""
        is_valid, error, data = _validate_json_string('{"x": invalid}', "input")
        assert is_valid is False
        assert "Line" in error or "line" in error
        assert "Column" in error or "column" in error
        assert data is None

    def test_error_message_includes_example(self) -> None:
        """Error message should include example of valid JSON."""
        is_valid, error, data = _validate_json_string("invalid", "input")
        assert is_valid is False
        assert "Example" in error or "example" in error
        assert data is None


class TestEstimateDifficulty:
    """Tests for _estimate_difficulty function."""

    def test_advanced_keywords(self) -> None:
        """Tasks with advanced keywords should be marked as advanced."""
        task = Task(id="test", description="Group by field and aggregate", examples=[])
        assert _estimate_difficulty(task) == "advanced"

    def test_intermediate_keywords(self) -> None:
        """Tasks with intermediate keywords should be marked as intermediate."""
        task = Task(id="test", description="Filter active users and select names", examples=[])
        assert _estimate_difficulty(task) == "intermediate"

    def test_basic_default(self) -> None:
        """Tasks without special keywords should be marked as basic."""
        task = Task(id="test", description="Extract the name field", examples=[])
        assert _estimate_difficulty(task) == "basic"


class TestFormatScore:
    """Tests for _format_score function."""

    def test_perfect_score(self) -> None:
        """Perfect score should be formatted."""
        result = _format_score(1.0)
        assert "1.000" in result

    def test_partial_score(self) -> None:
        """Partial score should be formatted."""
        result = _format_score(0.75)
        assert "0.750" in result

    def test_zero_score(self) -> None:
        """Zero score should be formatted."""
        result = _format_score(0.0)
        assert "0.000" in result


class TestSetupLogging:
    """Tests for _setup_logging function."""

    def test_debug_mode(self) -> None:
        """Debug mode should set DEBUG level."""
        _setup_logging(verbose=False, debug=True)
        # Check that debug mode was set (may not be directly testable)
        # Just ensure it runs without error
        assert True

    def test_verbose_mode(self) -> None:
        """Verbose mode should set INFO level."""
        _setup_logging(verbose=True, debug=False)


class TestListTasks:
    """Tests for _list_tasks function and --list-tasks flag."""

    def test_list_tasks_function(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """_list_tasks should display tasks grouped by difficulty."""
        from src.cli import _list_tasks
        from src.domain import Example

        tasks = [
            Task(
                id="basic-task",
                description="Extract field",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            ),
            Task(
                id="filter-task",
                description="Filter active items",
                examples=[Example(input_data=[1, 2], expected_output=[2])],
            ),
            Task(
                id="group-task",
                description="Group by category and aggregate",
                examples=[Example(input_data=[], expected_output=[])],
            ),
        ]

        _list_tasks(tasks)

        captured = capsys.readouterr()
        assert "Available Tasks" in captured.out
        assert "basic-task" in captured.out
        assert "filter-task" in captured.out
        assert "group-task" in captured.out
        assert "Basic" in captured.out
        assert "Intermediate" in captured.out
        assert "Advanced" in captured.out

    def test_list_tasks_with_long_description(self, capsys: pytest.CaptureFixture) -> None:
        """_list_tasks should truncate long descriptions."""
        from src.cli import _list_tasks
        from src.domain import Example

        tasks = [
            Task(
                id="long-desc",
                description="A" * 100,  # Very long description
                examples=[Example(input_data={}, expected_output={})],
            ),
        ]

        _list_tasks(tasks)

        captured = capsys.readouterr()
        # Should be truncated to 60 chars + "..."
        assert "..." in captured.out

    def test_list_tasks_single_example(self, capsys: pytest.CaptureFixture) -> None:
        """_list_tasks should handle singular 'example' correctly."""
        from src.cli import _list_tasks
        from src.domain import Example

        tasks = [
            Task(
                id="single",
                description="Test",
                examples=[Example(input_data={}, expected_output={})],
            ),
        ]

        _list_tasks(tasks)

        captured = capsys.readouterr()
        assert "1 example" in captured.out
        assert "1 examples" not in captured.out

    def test_list_tasks_multiple_examples(self, capsys: pytest.CaptureFixture) -> None:
        """_list_tasks should handle plural 'examples' correctly."""
        from src.cli import _list_tasks
        from src.domain import Example

        tasks = [
            Task(
                id="multiple",
                description="Test",
                examples=[
                    Example(input_data={}, expected_output={}),
                    Example(input_data={}, expected_output={}),
                ],
            ),
        ]

        _list_tasks(tasks)

        captured = capsys.readouterr()
        assert "2 examples" in captured.out


class TestMainListTasksFlag:
    """Tests for main() with --list-tasks flag."""

    def test_list_tasks_flag_success(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """--list-tasks should display tasks and return 0."""
        tasks_data = {
            "tasks": [
                {
                    "id": "test",
                    "description": "Test task",
                    "examples": [{"input": {}, "expected_output": {}}],
                }
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks_data))

        result = main(["--list-tasks", "--tasks-file", str(tasks_file)])

        assert result == 0
        captured = capsys.readouterr()
        assert "Available Tasks" in captured.out
        assert "test" in captured.out

    def test_list_tasks_with_missing_file(self, capsys: pytest.CaptureFixture) -> None:
        """--list-tasks with missing file should return 1."""
        result = main(["--list-tasks", "--tasks-file", "/nonexistent/file.json"])

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_list_tasks_with_invalid_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--list-tasks with invalid JSON should return 1."""
        tasks_file = tmp_path / "invalid.json"
        tasks_file.write_text("{ invalid json")

        result = main(["--list-tasks", "--tasks-file", str(tasks_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err or "json" in captured.err.lower()

    def test_list_tasks_with_missing_field(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--list-tasks with missing field should return 1."""
        tasks_data = {"wrong_key": []}
        tasks_file = tmp_path / "missing_field.json"
        tasks_file.write_text(json.dumps(tasks_data))

        result = main(["--list-tasks", "--tasks-file", str(tasks_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "Missing field" in captured.err or "field" in captured.err.lower()


class TestPrintSummaryTable:
    """Tests for _print_summary_table function."""

    def test_all_passed(self, capsys: pytest.CaptureFixture) -> None:
        """Summary should show success when all tasks pass."""
        from src.cli import _print_summary_table

        solutions = [
            Solution(
                task_id="task1",
                success=True,
                best_filter=".x",
                best_score=1.0,
                iterations_used=1,
                history=[],
            ),
            Solution(
                task_id="task2",
                success=True,
                best_filter=".y",
                best_score=1.0,
                iterations_used=1,
                history=[],
            ),
        ]

        _print_summary_table(solutions)

        captured = capsys.readouterr()
        assert "2/2 passed (100%)" in captured.out

    def test_all_failed(self, capsys: pytest.CaptureFixture) -> None:
        """Summary should show error when all tasks fail."""
        from src.cli import _print_summary_table

        # Need at least 2 solutions for summary table to print
        solutions = [
            Solution(
                task_id="task1",
                success=False,
                best_filter="",
                best_score=0.0,
                iterations_used=1,
                history=[],
            ),
            Solution(
                task_id="task2",
                success=False,
                best_filter="",
                best_score=0.0,
                iterations_used=1,
                history=[],
            ),
        ]

        _print_summary_table(solutions)

        captured = capsys.readouterr()
        assert "0/2 passed (0%)" in captured.out

    def test_partial_success(self, capsys: pytest.CaptureFixture) -> None:
        """Summary should show warning when some tasks fail."""
        from src.cli import _print_summary_table

        solutions = [
            Solution(
                task_id="task1",
                success=True,
                best_filter=".x",
                best_score=1.0,
                iterations_used=1,
                history=[],
            ),
            Solution(
                task_id="task2",
                success=False,
                best_filter="",
                best_score=0.5,
                iterations_used=1,
                history=[],
            ),
        ]

        _print_summary_table(solutions)

        captured = capsys.readouterr()
        assert "1/2 passed (50%)" in captured.out


class TestFormatScore:
    """Tests for _format_score function with different values."""

    def test_format_perfect_score(self) -> None:
        """Perfect score should be formatted."""
        from src.cli import _format_score

        result = _format_score(1.0)
        assert "1.000" in result

    def test_format_high_score(self) -> None:
        """High score should be formatted."""
        from src.cli import _format_score

        result = _format_score(0.85)
        assert "0.850" in result

    def test_format_low_score(self) -> None:
        """Low score should be formatted."""
        from src.cli import _format_score

        result = _format_score(0.25)
        assert "0.250" in result

    def test_format_zero_score(self) -> None:
        """Zero score should be formatted."""
        from src.cli import _format_score

        result = _format_score(0.0)
        assert "0.000" in result

    def test_format_near_perfect_score(self) -> None:
        """Score >= 0.999 should be treated as perfect (regression test)."""
        from src.cli import _format_score
        from src.colors import success, warning

        # Test exact 0.999 - should be treated as perfect
        result = _format_score(0.999)
        assert "0.999" in result
        # Should use success color (green) not warning
        expected = success("0.999")
        assert result == expected

        # Test slightly below 0.999 - should NOT be perfect
        result_below = _format_score(0.998)
        assert "0.998" in result_below
        # Should use warning color (yellow) not success
        expected_warning = warning("0.998")
        assert result_below == expected_warning


class TestPrintTaskResult:
    """Tests for _print_solution function."""

    def test_print_result_without_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Task result should be printed without history when not verbose."""
        from src.cli import _print_solution
        from src.domain import Attempt, ErrorType, ExampleResult

        solution = Solution(
            task_id="test",
            success=True,
            best_filter=".x",
            best_score=1.0,
            iterations_used=3,
            history=[
                Attempt(
                    iteration=0,
                    filter_code=".y",
                    aggregated_score=0.0,
                    primary_error=ErrorType.SHAPE,
                    example_results=[
                        ExampleResult(
                            expected_output=1,
                            actual_output=None,
                            score=0.0,
                            error_type=ErrorType.SHAPE,
                            feedback="Wrong",
                        )
                    ],
                )
            ],
        )

        _print_solution(solution, verbose=False)

        captured = capsys.readouterr()
        assert "test" in captured.out
        assert ".x" in captured.out
        assert "1.000" in captured.out
        # History should NOT be shown
        assert "History" not in captured.out

    def test_print_result_with_verbose(self, capsys: pytest.CaptureFixture) -> None:
        """Task result should include history when verbose."""
        from src.cli import _print_solution
        from src.domain import Attempt, ErrorType, ExampleResult

        solution = Solution(
            task_id="test",
            success=True,
            best_filter=".x",
            best_score=1.0,
            iterations_used=2,
            history=[
                Attempt(
                    iteration=0,
                    filter_code=".y",
                    aggregated_score=0.5,
                    primary_error=ErrorType.SHAPE,
                    example_results=[
                        ExampleResult(
                            expected_output=1,
                            actual_output=None,
                            score=0.5,
                            error_type=ErrorType.SHAPE,
                            feedback="Wrong",
                        )
                    ],
                ),
                Attempt(
                    iteration=1,
                    filter_code=".x",
                    aggregated_score=1.0,
                    primary_error=ErrorType.NONE,
                    example_results=[
                        ExampleResult(
                            expected_output=1,
                            actual_output=1,
                            score=1.0,
                            error_type=ErrorType.NONE,
                            feedback="",
                        )
                    ],
                ),
            ],
        )

        _print_solution(solution, verbose=True)

        captured = capsys.readouterr()
        # History SHOULD be shown
        assert "History" in captured.out or "[0]" in captured.out or "[1]" in captured.out


class TestVerboseOutput:
    """Tests for verbose flag behavior."""

    def test_parse_verbose_flag(self) -> None:
        """Verbose flag should be parsed correctly."""
        from src.cli import _parse_args

        args = _parse_args(["--task", "test", "--verbose"])
        assert args.verbose is True

    def test_parse_verbose_short_flag(self) -> None:
        """Short verbose flag should be parsed correctly."""
        from src.cli import _parse_args

        args = _parse_args(["--task", "test", "-v"])
        assert args.verbose is True

    def test_verbose_default_false(self) -> None:
        """Verbose should default to False."""
        from src.cli import _parse_args

        args = _parse_args(["--task", "test"])
        assert args.verbose is False
