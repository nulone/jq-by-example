"""
Main iteration loop with anti-stuck protocol - coordinates generator and reviewer.

This module provides the Orchestrator class that manages the iterative refinement
loop for jq filter synthesis, coordinating between the generator and reviewer
components while implementing anti-stuck mechanisms.
"""

import logging
from dataclasses import replace

from src.domain import Attempt, Solution, Task
from src.generator import JQGenerator
from src.reviewer import AlgorithmicReviewer

logger = logging.getLogger(__name__)


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

            # Generate a candidate filter
            try:
                filter_code = self.generator.generate(task, list(history) if history else None)
            except Exception as e:
                if verbose:
                    logger.warning("Generator failed on iteration %d: %s", iteration, e)
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    logger.info("Stagnation limit reached after generator failure")
                    break
                continue

            # Check for duplicates (normalized comparison)
            normalized = self._normalize(filter_code)
            if normalized in seen_filters:
                logger.debug("Duplicate filter detected: '%s'", filter_code)
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    logger.info("Stagnation limit reached due to duplicate filters")
                    break
                continue

            seen_filters.add(normalized)

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

        Normalization involves removing all whitespace while preserving case.
        This allows detecting semantically identical filters like '.foo' and '. foo'
        while respecting case-sensitivity of jq field names.

        Args:
            filter_code: The filter code to normalize.

        Returns:
            Normalized filter string for comparison.
        """
        return "".join(filter_code.split())
