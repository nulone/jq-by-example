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

from src.cli import _create_interactive_task, _parse_args, load_tasks, main
from src.domain import Task


class TestLoadTasksValidJSON:
    """Tests for load_tasks parsing valid JSON files."""

    def test_parses_single_task(self, tmp_path: Path):
        """Single task is correctly parsed into Task object."""
        tasks_data = {
            "tasks": [
                {
                    "id": "test-task",
                    "description": "Extract the name field",
                    "examples": [
                        {"input": {"name": "Alice"}, "expected_output": "Alice"}
                    ],
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

        assert task.examples[0].input_data == {
            "user": {"name": "Alice", "roles": ["admin"]}
        }
        assert task.examples[0].expected_output == {
            "name": "Alice",
            "roles": ["admin"],
        }

    def test_creates_task_with_array_input(self):
        """Interactive mode handles array input."""
        task = _create_interactive_task(
            input_json='[1, 2, 3]',
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

                main(
                    ["--task", "test", "--tasks-file", str(tasks_file), "--max-iters", "7"]
                )

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
