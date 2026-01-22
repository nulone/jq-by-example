"""
Command-line interface for running JQ-Synth in batch or interactive mode.

This module provides the CLI entry point for the JQ-Synth tool, supporting
both batch processing of task files and interactive single-example synthesis.
"""

import argparse
import json
import logging
import sys
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from src.colors import bold, cyan, dim, error, info, success, warning
from src.domain import Example, Solution, Task
from src.executor import JQExecutor
from src.generator import GenerationError, JQGenerator
from src.orchestrator import Orchestrator
from src.reviewer import AlgorithmicReviewer

logger = logging.getLogger(__name__)


def load_tasks(path: str) -> list[Task]:
    """
    Load tasks from a JSON file.

    Args:
        path: Path to the JSON file containing task definitions.

    Returns:
        List of Task objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields are missing from the task definitions.
    """
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: list[Task] = []

    for task_data in data["tasks"]:
        examples = [
            Example(
                input_data=ex["input"],
                expected_output=ex["expected_output"],
            )
            for ex in task_data["examples"]
        ]

        task = Task(
            id=task_data["id"],
            description=task_data["description"],
            examples=examples,
        )
        tasks.append(task)

    logger.debug("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def _format_jq_not_found_error() -> str:
    """Format helpful error message when jq binary is not found."""
    return f"""
{error("⚠️  jq binary not found in your PATH")}

jq-synth requires the jq command-line tool to be installed.

{bold("Quick setup:")}
  • macOS:    {info("brew install jq")}
  • Ubuntu:   {info("sudo apt-get install jq")}
  • Windows:  {info("choco install jq")}

{bold("After installation:")}
  1. Verify: {cyan("jq --version")}
  2. Try again: {cyan("jq-synth --help")}

{dim("Need help? https://stedolan.github.io/jq/download/")}
"""


def _format_api_key_error(provider: str) -> str:
    """Format helpful error message when API key is missing."""
    if provider == "openai":
        return f"""
{error("🔑 API key required")}

jq-synth uses AI to generate jq filters. You need an OpenAI API key.

{bold("Quick setup:")}
  1. Sign up: {info("https://platform.openai.com")}
  2. Create key: {info("https://platform.openai.com/api-keys")}
  3. Set environment variable:
     {cyan("export OPENAI_API_KEY='sk-...'")}

{bold("Alternative providers:")}
  • Anthropic: {cyan("jq-synth --provider anthropic")}
  • Local (free): See README for Ollama setup

{dim("Tip: Add the export to your ~/.bashrc or ~/.zshrc for persistence")}
"""
    elif provider == "anthropic":
        return f"""
{error("🔑 API key required")}

{bold("Anthropic API setup:")}
  1. Sign up: {info("https://console.anthropic.com")}
  2. Create key: {info("https://console.anthropic.com/settings/keys")}
  3. Set environment variable:
     {cyan("export ANTHROPIC_API_KEY='sk-ant-...'")}

{dim("Tip: OpenAI is the default provider: jq-synth --provider openai")}
"""
    else:
        return f"{error('API key required')} for provider: {provider}"


def _format_task_not_found_error(task_id: str, available_tasks: list[Task]) -> str:
    """Format helpful error when task is not found."""
    available_ids = [t.id for t in available_tasks]
    close_matches = get_close_matches(task_id, available_ids, n=1, cutoff=0.6)

    msg = f"\n{error('⚠️  Task not found:')} {bold(task_id)}\n"

    if close_matches:
        match = close_matches[0]
        matched_task = next(t for t in available_tasks if t.id == match)
        msg += f"\n{warning('Did you mean:')} {cyan(match)}\n"
        msg += f"{dim(matched_task.description)}\n"

    msg += f"\n{bold('Available tasks:')}\n"
    for task in available_tasks[:5]:  # Show first 5
        msg += f"  • {cyan(task.id):<20} {task.description[:50]}\n"

    if len(available_tasks) > 5:
        msg += f"\n{dim(f'... and {len(available_tasks) - 5} more')}\n"

    msg += f"\n{dim('View all: jq-synth --list-tasks')}\n"

    return msg


def _validate_json_string(json_str: str, param_name: str) -> tuple[bool, str, Any]:
    """
    Validate a JSON string and provide helpful error messages.

    Args:
        json_str: The JSON string to validate.
        param_name: Name of the parameter (for error messages).

    Returns:
        Tuple of (is_valid, error_message, parsed_data).
    """
    try:
        data = json.loads(json_str)
        return True, "", data
    except json.JSONDecodeError as e:
        # Build helpful error message
        lines = json_str.split("\n")
        error_line = lines[e.lineno - 1] if e.lineno <= len(lines) else ""

        # Detect common issues
        suggestions = []
        if "{" in json_str and "}" not in json_str:
            suggestions.append("Missing closing brace }")
        if "[" in json_str and "]" not in json_str:
            suggestions.append("Missing closing bracket ]")
        if json_str.count('"') % 2 != 0:
            suggestions.append('Unmatched quote "')
        if "'" in json_str and '"' not in json_str:
            suggestions.append("Use double quotes \" instead of single quotes '")

        msg = f"""
{error("Invalid JSON")} in --{param_name}

{error("Error:")} {e.msg}
{error("Position:")} Line {e.lineno}, Column {e.colno}

{error_line}
{" " * (e.colno - 1)}^

{bold("Common issues:")}
"""
        for s in suggestions[:3]:  # Limit to 3 suggestions
            msg += f"  • {s}\n"

        msg += f"""
{info("Example of valid JSON:")}
  jq-synth -i '{{"x": 1}}' -o '1' -d 'Extract x'

{dim("Tip: Use a JSON validator: https://jsonlint.com")}
"""
        return False, msg, None


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Optional list of arguments. If None, uses sys.argv.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="jq-synth",
        description="AI-Powered JQ Filter Synthesis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific task
  jq-synth --task nested-field

  # Run all tasks from a file
  jq-synth --task all --tasks-file data/tasks.json

  # Interactive mode
  jq-synth --input '{"x": 1}' --output '1' --desc 'Extract x'

  # Baseline (single-shot) mode
  jq-synth --task nested-field --baseline
""",
    )

    # Task selection
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task ID to run, or 'all' to run all tasks",
    )

    parser.add_argument(
        "--tasks-file",
        type=str,
        default="data/tasks.json",
        help="Path to tasks JSON file (default: data/tasks.json)",
    )

    # Iteration control
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum iterations per task (default: 10)",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Single-shot mode (max_iterations=1)",
    )

    # Interactive mode
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input JSON for interactive mode",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Expected output JSON for interactive mode",
    )

    parser.add_argument(
        "-d",
        "--desc",
        type=str,
        default="Transform the input to produce the expected output",
        help="Task description for interactive mode",
    )

    # LLM provider configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        help="LLM provider type (default: from LLM_PROVIDER env or 'openai')",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model identifier (default: from LLM_MODEL env or provider default)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for OpenAI-compatible providers (default: from LLM_BASE_URL env)",
    )

    # Output control
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (shows iteration details)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows detailed internal state)",
    )

    # Task management
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    return parser.parse_args(args)


def _setup_logging(verbose: bool, debug: bool) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: If True, set level to INFO; otherwise WARNING.
        debug: If True, set level to DEBUG (overrides verbose).
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_interactive_task(
    input_json: str,
    output_json: str,
    description: str,
) -> Task:
    """
    Create a Task from interactive mode arguments.

    Args:
        input_json: JSON string for input data.
        output_json: JSON string for expected output.
        description: Task description.

    Returns:
        Task with a single example.

    Raises:
        json.JSONDecodeError: If JSON strings are invalid.
    """
    input_data: Any = json.loads(input_json)
    expected_output: Any = json.loads(output_json)

    example = Example(input_data=input_data, expected_output=expected_output)

    return Task(
        id="interactive",
        description=description,
        examples=[example],
    )


def _format_score(score: float) -> str:
    """Format score with color based on value."""
    if score >= 0.999:  # Perfect score per domain model (Attempt.is_perfect)
        return success(f"{score:.3f}")
    elif score >= 0.8:
        return warning(f"{score:.3f}")
    else:
        return error(f"{score:.3f}")


def _print_solution(solution: Solution, verbose: bool = False) -> None:
    """
    Print a solution to stdout.

    Args:
        solution: The solution to print.
        verbose: If True, print additional details.
    """
    status = success("✓") if solution.success else error("✗")
    print(f"\n{status} Task: {bold(solution.task_id)}")
    print(f"  Filter: {cyan(solution.best_filter)}")
    print(f"  Score: {_format_score(solution.best_score)}")
    print(f"  Iterations: {solution.iterations_used}")

    if verbose and solution.history:
        print(f"  {dim('History:')}")
        for attempt in solution.history:
            score_str = _format_score(attempt.aggregated_score)
            print(
                f"    {dim(f'[{attempt.iteration}]')} score={score_str} "
                f"error={attempt.primary_error.value} filter='{dim(attempt.filter_code)}'"
            )


def _estimate_difficulty(task: Task) -> str:
    """Estimate task difficulty based on heuristics."""
    desc_lower = task.description.lower()

    if any(word in desc_lower for word in ["group", "aggregate", "reduce", "sum all"]):
        return "advanced"
    elif any(word in desc_lower for word in ["filter", "select", "extract multiple", "skip"]):
        return "intermediate"
    else:
        return "basic"


def _list_tasks(tasks: list[Task]) -> None:
    """
    Display all available tasks in a formatted table.

    Args:
        tasks: List of Task objects to display.
    """
    print(f"\n{bold(f'Available Tasks ({len(tasks)})')}\n")

    # Group by difficulty
    basic = []
    intermediate = []
    advanced = []

    for task in tasks:
        difficulty = _estimate_difficulty(task)
        if difficulty == "basic":
            basic.append(task)
        elif difficulty == "intermediate":
            intermediate.append(task)
        else:
            advanced.append(task)

    def print_group(name: str, tasks_list: list[Task]) -> None:
        if not tasks_list:
            return
        print(f"{bold(name)} ({len(tasks_list)}):")
        for task in tasks_list:
            examples_text = f"{len(task.examples)} example"
            if len(task.examples) != 1:
                examples_text += "s"
            # Truncate description if too long
            desc = task.description[:60]
            if len(task.description) > 60:
                desc += "..."
            print(f"  • {cyan(f'{task.id:<20}')} {desc} ({dim(examples_text)})")
        print()

    print_group("Basic", basic)
    print_group("Intermediate", intermediate)
    print_group("Advanced", advanced)

    print(dim("Usage: jq-synth --task <task-id>"))
    print(dim("Run all: jq-synth --task all"))
    print()


def _print_summary_table(solutions: list[Solution]) -> None:
    """
    Print a summary table for multiple solutions.

    Args:
        solutions: List of solutions to summarize.
    """
    if len(solutions) <= 1:
        return

    print("\n" + "=" * 60)
    print(bold("Summary"))
    print("=" * 60)

    # Header
    print(f"{'Task ID':<25} {'Status':<10} {'Score':<10} {'Iters':<10}")
    print("-" * 60)

    # Rows
    for sol in solutions:
        if sol.success:
            status = success("PASS")
        else:
            status = error("FAIL")
        score_str = _format_score(sol.best_score)
        print(f"{sol.task_id:<25} {status:<20} {score_str:<20} {sol.iterations_used:<10}")

    # Footer
    print("-" * 60)
    passed = sum(1 for s in solutions if s.success)
    total = len(solutions)
    pass_rate = 100 * passed / total if total > 0 else 0
    if passed == total:
        summary_str = success(f"{passed}/{total} passed ({pass_rate:.0f}%)")
    elif passed == 0:
        summary_str = error(f"{passed}/{total} passed ({pass_rate:.0f}%)")
    else:
        summary_str = warning(f"{passed}/{total} passed ({pass_rate:.0f}%)")
    print(f"Total: {summary_str}")


def main(args: list[str] | None = None) -> int:
    """
    CLI entry point for JQ-Synth.

    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv.

    Returns:
        0 if all tasks succeed, 1 otherwise.
    """
    parsed = _parse_args(args)
    _setup_logging(parsed.verbose, parsed.debug)

    # Handle --list-tasks flag
    if parsed.list_tasks:
        try:
            all_tasks = load_tasks(parsed.tasks_file)
            _list_tasks(all_tasks)
            return 0
        except FileNotFoundError:
            print(error(f"Error: Tasks file not found: {parsed.tasks_file}"), file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(error(f"Error: Invalid JSON in tasks file: {e}"), file=sys.stderr)
            return 1
        except KeyError as e:
            print(error(f"Error: Missing field in tasks file: {e}"), file=sys.stderr)
            return 1

    # Determine mode: interactive or batch
    is_interactive = parsed.input is not None and parsed.output is not None

    if is_interactive:
        # Interactive mode with JSON validation
        valid, err_msg, input_data = _validate_json_string(parsed.input, "input")
        if not valid:
            print(err_msg, file=sys.stderr)
            return 1

        valid, err_msg, output_data = _validate_json_string(parsed.output, "output")
        if not valid:
            print(err_msg, file=sys.stderr)
            return 1

        example = Example(input_data=input_data, expected_output=output_data)
        task = Task(id="interactive", description=parsed.desc, examples=[example])
        tasks = [task]
    else:
        # Batch mode - need --task argument
        if not parsed.task:
            print(
                error("Error: Must specify --task or use interactive mode (--input and --output)"),
                file=sys.stderr,
            )
            print(info("\nExamples:"))
            print(f"  {cyan('jq-synth --task nested-field')}")
            print(
                f"  {cyan('jq-synth -i')} "
                + "'{\"x\": 1}' "
                + f"{cyan('-o')} '1' {cyan('-d')} 'Extract x'\n"
            )
            return 1

        # Load tasks from file
        try:
            all_tasks = load_tasks(parsed.tasks_file)
        except FileNotFoundError:
            print(error(f"Error: Tasks file not found: {parsed.tasks_file}"), file=sys.stderr)
            print(info(f"Expected location: {Path(parsed.tasks_file).absolute()}"))
            return 1
        except json.JSONDecodeError as e:
            print(error(f"Error: Invalid JSON in tasks file: {e}"), file=sys.stderr)
            return 1
        except KeyError as e:
            print(error(f"Error: Missing field in tasks file: {e}"), file=sys.stderr)
            return 1

        # Filter tasks
        if parsed.task.lower() == "all":
            tasks = all_tasks
        else:
            tasks = [t for t in all_tasks if t.id == parsed.task]
            if not tasks:
                print(_format_task_not_found_error(parsed.task, all_tasks), file=sys.stderr)
                return 1

    # Initialize components
    try:
        executor = JQExecutor()
    except RuntimeError as e:
        if "jq binary not found" in str(e) or "not found in PATH" in str(e):
            print(_format_jq_not_found_error(), file=sys.stderr)
        else:
            print(error(f"Error: {e}"), file=sys.stderr)
        return 1

    try:
        generator = JQGenerator(
            provider_type=parsed.provider,
            model=parsed.model,
            base_url=parsed.base_url,
        )
    except ValueError as e:
        error_str = str(e).lower()
        if "api key" in error_str or "api_key" in error_str:
            provider = parsed.provider or "openai"
            print(_format_api_key_error(provider), file=sys.stderr)
        else:
            print(error(f"Error: {e}"), file=sys.stderr)
        return 1

    reviewer = AlgorithmicReviewer(executor)

    # Determine max iterations
    max_iterations = 1 if parsed.baseline else parsed.max_iters

    orchestrator = Orchestrator(
        generator=generator,
        reviewer=reviewer,
        max_iterations=max_iterations,
    )

    # Run tasks
    solutions: list[Solution] = []
    total_time_sec = 0.0

    for task_num, task in enumerate(tasks, 1):
        print(f"\n{'=' * 60}")
        print(f"[{task_num}/{len(tasks)}] Solving: {task.id}")
        print(f"Description: {task.description}")
        print(f"Examples: {len(task.examples)}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            solution = orchestrator.solve(task, verbose=parsed.verbose)
            solutions.append(solution)

            elapsed = time.time() - start_time
            total_time_sec += elapsed

            _print_solution(solution, verbose=parsed.verbose)
            print(f"  Time: {elapsed:.2f}s")

        except GenerationError as e:
            elapsed = time.time() - start_time
            total_time_sec += elapsed

            logger.error("Generation failed for task %s: %s", task.id, e)
            print(f"\n✗ Error: {e}")

            # Create a failed solution
            solutions.append(
                Solution(
                    task_id=task.id,
                    success=False,
                    best_filter="",
                    best_score=0.0,
                    iterations_used=0,
                    history=[],
                )
            )
            _print_solution(solutions[-1], verbose=parsed.verbose)
            print(f"  Time: {elapsed:.2f}s")

    # Print summary for multi-task runs
    _print_summary_table(solutions)

    # Print overall summary
    if solutions:
        print(f"\n{'=' * 60}")
        print(bold("OVERALL SUMMARY"))
        print(f"{'=' * 60}")
        passed = sum(1 for s in solutions if s.success)
        total = len(solutions)
        pass_rate = 100 * passed / total if total > 0 else 0

        if passed == total:
            tasks_str = success(f"{passed}/{total} passed ({pass_rate:.1f}%)")
        elif passed == 0:
            tasks_str = error(f"{passed}/{total} passed ({pass_rate:.1f}%)")
        else:
            tasks_str = warning(f"{passed}/{total} passed ({pass_rate:.1f}%)")

        print(f"Tasks: {tasks_str}")
        print(f"Total time: {cyan(f'{total_time_sec:.2f}s')}")
        if total_time_sec > 0:
            print(f"Average time per task: {cyan(f'{total_time_sec / total:.2f}s')}")
        print(f"{'=' * 60}")

    # Return code
    all_success = all(s.success for s in solutions)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
