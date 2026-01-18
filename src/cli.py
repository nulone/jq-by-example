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
from pathlib import Path
from typing import Any

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


def _print_solution(solution: Solution, verbose: bool = False) -> None:
    """
    Print a solution to stdout.

    Args:
        solution: The solution to print.
        verbose: If True, print additional details.
    """
    status = "✓" if solution.success else "✗"
    print(f"\n{status} Task: {solution.task_id}")
    print(f"  Filter: {solution.best_filter}")
    print(f"  Score: {solution.best_score:.3f}")
    print(f"  Iterations: {solution.iterations_used}")

    if verbose and solution.history:
        print("  History:")
        for attempt in solution.history:
            print(
                f"    [{attempt.iteration}] score={attempt.aggregated_score:.3f} "
                f"error={attempt.primary_error.value} filter='{attempt.filter_code}'"
            )


def _print_summary_table(solutions: list[Solution]) -> None:
    """
    Print a summary table for multiple solutions.

    Args:
        solutions: List of solutions to summarize.
    """
    if len(solutions) <= 1:
        return

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Header
    print(f"{'Task ID':<25} {'Status':<10} {'Score':<10} {'Iters':<10}")
    print("-" * 60)

    # Rows
    for sol in solutions:
        status = "PASS" if sol.success else "FAIL"
        print(f"{sol.task_id:<25} {status:<10} {sol.best_score:<10.3f} {sol.iterations_used:<10}")

    # Footer
    print("-" * 60)
    passed = sum(1 for s in solutions if s.success)
    total = len(solutions)
    print(f"Total: {passed}/{total} passed")


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

    # Determine mode: interactive or batch
    is_interactive = parsed.input is not None and parsed.output is not None

    if is_interactive:
        # Interactive mode
        try:
            task = _create_interactive_task(
                parsed.input,
                parsed.output,
                parsed.desc,
            )
            tasks = [task]
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input or output: {e}", file=sys.stderr)
            return 1
    else:
        # Batch mode - need --task argument
        if not parsed.task:
            print(
                "Error: Must specify --task or use interactive mode (--input and --output)",
                file=sys.stderr,
            )
            return 1

        # Load tasks from file
        try:
            all_tasks = load_tasks(parsed.tasks_file)
        except FileNotFoundError:
            print(f"Error: Tasks file not found: {parsed.tasks_file}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in tasks file: {e}", file=sys.stderr)
            return 1
        except KeyError as e:
            print(f"Error: Missing field in tasks file: {e}", file=sys.stderr)
            return 1

        # Filter tasks
        if parsed.task.lower() == "all":
            tasks = all_tasks
        else:
            tasks = [t for t in all_tasks if t.id == parsed.task]
            if not tasks:
                print(f"Error: Task not found: {parsed.task}", file=sys.stderr)
                print(f"Available tasks: {[t.id for t in all_tasks]}", file=sys.stderr)
                return 1

    # Initialize components
    try:
        executor = JQExecutor()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        generator = JQGenerator(
            provider_type=parsed.provider,
            model=parsed.model,
            base_url=getattr(parsed, "base_url", None),
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
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
        print("OVERALL SUMMARY")
        print(f"{'=' * 60}")
        passed = sum(1 for s in solutions if s.success)
        total = len(solutions)
        print(f"Tasks: {passed}/{total} passed ({100 * passed / total:.1f}%)")
        print(f"Total time: {total_time_sec:.2f}s")
        if total_time_sec > 0:
            print(f"Average time per task: {total_time_sec / total:.2f}s")
        print(f"{'=' * 60}")

    # Return code
    all_success = all(s.success for s in solutions)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
