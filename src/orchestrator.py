"""
Main iteration loop with anti-stuck protocol - coordinates generator and reviewer.

This module provides the Orchestrator class that manages the iterative refinement
loop for jq filter synthesis, coordinating between the generator and reviewer
components while implementing anti-stuck mechanisms.
"""

import logging
import sys
from dataclasses import replace

from src.colors import dim, error, success, warning
from src.domain import Attempt, Solution, Task
from src.generator import JQGenerator
from src.reviewer import AlgorithmicReviewer

logger = logging.getLogger(__name__)


def _should_show_progress() -> bool:
    """Check if progress indicator should be displayed."""
    return sys.stdout.isatty() and not logger.isEnabledFor(logging.DEBUG)


def _print_progress(iteration: int, max_iter: int, status: str, clear_line: bool = False) -> None:
    """
    Print progress indicator for current iteration.

    Args:
        iteration: Current iteration number.
        max_iter: Maximum iterations.
        status: Status message to display.
        clear_line: If True, clear the line before printing.
    """
    if not _should_show_progress():
        return

    prefix = f"Iteration {iteration}/{max_iter}"
    if clear_line:
        # Clear line and return to start
        print(f"\r{' ' * 80}\r{prefix}  {status}", end="", flush=True)
    else:
        print(f"\r{prefix}  {status}", end="", flush=True)


def _print_progress_done(message: str) -> None:
    """
    Print final progress message and move to new line.

    Args:
        message: Final message to display.
    """
    if not _should_show_progress():
        return

    print(f"\r{' ' * 80}\r{message}")


class Orchestrator:
    """
    Coordinates the iterative jq filter synthesis loop.

    This class manages the generate-evaluate-refine cycle, tracking attempt history,
    detecting stagnation, and implementing anti-stuck protocols to ensure the
    synthesis process terminates with the best found solution.

    Attributes:
        generator: The JQGenerator instance for creating filter candidates.
        reviewer: The AlgorithmicReviewer instance for evaluating filters.
        max_iterations: Maximum number of generation attempts.
        stagnation_limit: Number of iterations without improvement before stopping.
    """

    def __init__(
        self,
        generator: JQGenerator,
        reviewer: AlgorithmicReviewer,
        max_iterations: int = 10,
        stagnation_limit: int = 3,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            generator: JQGenerator instance for creating filter candidates.
            reviewer: AlgorithmicReviewer instance for evaluating filters.
            max_iterations: Maximum number of generation attempts. Defaults to 10.
            stagnation_limit: Number of iterations without improvement before stopping.
                Defaults to 3.
        """
        self.generator = generator
        self.reviewer = reviewer
        self.max_iterations = max_iterations
        self.stagnation_limit = stagnation_limit

        logger.debug(
            "Orchestrator initialized: max_iterations=%d, stagnation_limit=%d",
            max_iterations,
            stagnation_limit,
        )

    def solve(self, task: Task, verbose: bool = False) -> Solution:
        """
        Attempt to synthesize a jq filter for the given task.

        Runs an iterative refinement loop that:
        1. Generates a candidate filter using the LLM
        2. Evaluates the filter against task examples
        3. Checks for success or stagnation
        4. Continues with feedback until solution found or limits reached

        Args:
            task: The task containing description and examples to solve.
            verbose: If True, logs additional information including errors.
                Defaults to False.

        Returns:
            Solution containing the best filter found, success status,
            and complete attempt history.
        """
        logger.info("Starting solve for task '%s'", task.id)

        history: list[Attempt] = []
        best: Attempt | None = None
        stagnation_counter = 0
        seen_filters: set[str] = set()

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Iteration %d/%d", iteration, self.max_iterations)

            # Show progress: generating filter
            _print_progress(
                iteration, self.max_iterations, "🤖 Generating filter...", clear_line=True
            )

            # Generate a candidate filter
            try:
                filter_code = self.generator.generate(task, list(history) if history else None)
            except Exception as e:
                if verbose:
                    logger.warning("Generator failed on iteration %d: %s", iteration, e)
                _print_progress_done(
                    f"{error('❌')} Iteration {iteration}/{self.max_iterations} - Generation failed"
                )
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    logger.info("Stagnation limit reached after generator failure")
                    break
                continue

            # Check for duplicates (normalized comparison)
            normalized = self._normalize(filter_code)
            if normalized in seen_filters:
                logger.debug("Duplicate filter detected: '%s'", filter_code)
                _print_progress_done(
                    f"{warning('⚠️')} Iteration {iteration}/{self.max_iterations} - Duplicate filter detected"
                )
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    logger.info("Stagnation limit reached due to duplicate filters")
                    break
                continue

            seen_filters.add(normalized)

            # Show progress: testing filter
            truncated_filter = filter_code[:50] + "..." if len(filter_code) > 50 else filter_code
            _print_progress(iteration, self.max_iterations, f"⚙️  Testing: {dim(truncated_filter)}")

            # Evaluate the filter
            attempt = self.reviewer.evaluate(task, filter_code)

            # Update iteration number (reviewer returns iteration=0)
            attempt = replace(attempt, iteration=iteration)
            history.append(attempt)

            logger.info(
                "Attempt %d: score=%.3f, is_perfect=%s, error=%s",
                iteration,
                attempt.aggregated_score,
                attempt.is_perfect,
                attempt.primary_error.value,
            )

            # Show progress: display score
            if attempt.is_perfect:
                score_display = success("✓ Score: 1.000 - Perfect match!")
            elif attempt.aggregated_score >= 0.8:
                score_display = warning(f"📊 Score: {attempt.aggregated_score:.3f}")
            else:
                score_display = error(f"📊 Score: {attempt.aggregated_score:.3f}")

            _print_progress_done(f"Iteration {iteration}/{self.max_iterations}  {score_display}")

            # Check for perfect solution
            if attempt.is_perfect:
                logger.info("Perfect solution found on iteration %d", iteration)
                return Solution(
                    task_id=task.id,
                    success=True,
                    best_filter=attempt.filter_code,
                    best_score=attempt.aggregated_score,
                    iterations_used=len(history),
                    history=history,
                )

            # Update best attempt and check for improvement
            if best is None or attempt.aggregated_score > best.aggregated_score:
                best = attempt
                stagnation_counter = 0
                logger.debug("New best score: %.3f", best.aggregated_score)
            else:
                stagnation_counter += 1
                logger.debug(
                    "No improvement, stagnation counter: %d/%d",
                    stagnation_counter,
                    self.stagnation_limit,
                )

            # Check stagnation limit after evaluation
            if stagnation_counter >= self.stagnation_limit:
                logger.info(
                    "Stagnation limit reached after %d iterations without improvement",
                    stagnation_counter,
                )
                break

        # Return best solution found (or failure if none)
        if best is not None:
            logger.info(
                "Solve completed: success=False, best_score=%.3f, iterations=%d",
                best.aggregated_score,
                len(history),
            )
            return Solution(
                task_id=task.id,
                success=False,
                best_filter=best.filter_code,
                best_score=best.aggregated_score,
                iterations_used=len(history),
                history=history,
            )

        # No attempts succeeded at all (all generator failures)
        logger.warning("No valid attempts made for task '%s'", task.id)
        return Solution(
            task_id=task.id,
            success=False,
            best_filter="",
            best_score=0.0,
            iterations_used=0,
            history=history,
        )

    def _normalize(self, filter_code: str) -> str:
        """
        Normalize a filter code for duplicate detection.

        Normalization removes whitespace OUTSIDE of string literals, while preserving:
        - String literal contents (spaces in "a b" are kept)
        - Case sensitivity of jq field names

        This allows detecting semantically identical filters like '.foo' and '. foo'
        while treating '.x == "a b"' and '.x == "ab"' as different filters.

        Args:
            filter_code: The filter code to normalize.

        Returns:
            Normalized filter string for comparison.

        Examples:
            >>> _normalize('.  x')
            '.x'
            >>> _normalize('.x == "a b"')
            '.x=="a b"'  # Space in string literal preserved
        """
        result = []
        in_string = False
        escape_next = False

        for char in filter_code:
            if escape_next:
                result.append(char)
                escape_next = False
                continue

            if char == "\\":
                result.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            if in_string:
                # Inside string literal - preserve everything including spaces
                result.append(char)
            elif not char.isspace():
                # Outside string literal - skip whitespace
                result.append(char)

        return "".join(result)
