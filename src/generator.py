"""
Generate jq filters using LLM providers.

This module provides the JQGenerator class that interfaces with various LLM APIs
(OpenAI, Anthropic, etc.) to generate jq filter expressions based on task
descriptions and input/output examples.
"""

import hashlib
import json
import logging
import re
import time

import httpx

from src.domain import Attempt, Task
from src.providers import LLMProvider, create_provider

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when filter generation fails."""

    pass


class JQGenerator:
    """
    Generates jq filters using LLM providers.

    This class interfaces with various LLM APIs (OpenAI, Anthropic, etc.) to generate
    jq filter expressions based on task descriptions and input/output examples. It
    supports iterative refinement by including previous attempt history in prompts.

    Attributes:
        provider: The LLM provider instance.
    """

    MAX_HISTORY_ATTEMPTS = 3
    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 1.0

    def __init__(
        self,
        provider: LLMProvider | None = None,
        provider_type: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the JQ generator.

        Args:
            provider: An LLMProvider instance. If provided, other arguments are ignored.
            provider_type: Provider type ('openai' or 'anthropic'). If not provided,
                reads from LLM_PROVIDER environment variable (default: 'openai').
            api_key: API key for the provider.
            model: Model identifier.
            base_url: Base URL (only for OpenAI-compatible providers).

        Raises:
            ValueError: If provider creation fails or required credentials are missing.
        """
        if provider is not None:
            self.provider = provider
        else:
            self.provider = create_provider(
                provider_type=provider_type,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )

        logger.debug("JQGenerator initialized with provider=%s", type(self.provider).__name__)

    def generate(self, task: Task, history: list[Attempt] | None = None) -> str:
        """
        Generate a jq filter for the given task.

        Args:
            task: The task containing description and input/output examples.
            history: Optional list of previous attempts for iterative refinement.
                Only the last 3 attempts are included in the prompt.

        Returns:
            A jq filter expression string.

        Raises:
            GenerationError: If the API call fails or returns an invalid response.
        """
        logger.info("Generating filter for task '%s'", task.id)

        prompt = self._build_prompt(task, history)

        # SECURITY: Log only prompt length and hash, never the actual content
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        logger.debug(
            "Built prompt: length=%d hash=%s",
            len(prompt),
            prompt_hash,
        )

        try:
            response_text = self._call_api_with_retry(prompt)
            filter_code = self._extract(response_text)

            logger.info("Generated filter: '%s'", filter_code)
            return filter_code

        except httpx.TimeoutException as e:
            logger.error("API request timed out: %s", e)
            raise GenerationError(
                f"API request timed out after {self.provider.TIMEOUT_SEC}s"
            ) from e

        except httpx.HTTPStatusError as e:
            logger.error("API returned error status: %s", e)
            raise GenerationError(f"API error: {e.response.status_code}") from e

        except httpx.RequestError as e:
            logger.error("API request failed: %s", e)
            raise GenerationError(f"API request failed: {e}") from e

        except RuntimeError as e:
            logger.error("Provider error: %s", e)
            raise GenerationError(f"Provider error: {e}") from e

    def _build_prompt(self, task: Task, history: list[Attempt] | None = None) -> str:
        """
        Build the user prompt for the API request.

        Args:
            task: The task to generate a filter for.
            history: Optional list of previous attempts.

        Returns:
            The formatted prompt string.
        """
        parts: list[str] = []

        # Task description
        parts.append(f"Task: {task.description}")
        parts.append("")

        # Examples
        for i, example in enumerate(task.examples, start=1):
            parts.append(f"Example {i}:")
            parts.append(f"Input: {json.dumps(example.input_data, sort_keys=True)}")
            parts.append(f"Expected Output: {json.dumps(example.expected_output, sort_keys=True)}")
            parts.append("")

        # Include history if provided (last N attempts)
        if history:
            recent_history = history[-self.MAX_HISTORY_ATTEMPTS :]

            parts.append("Previous attempts that did not fully succeed:")
            parts.append("")

            for attempt in recent_history:
                parts.append(f"- Filter: {attempt.filter_code}")
                parts.append(f"  Score: {attempt.aggregated_score:.2f}")
                parts.append(f"  Error Type: {attempt.primary_error.value}")

                # Include feedback from first failing example
                for result in attempt.example_results:
                    if result.score < 1.0:
                        parts.append(f"  Feedback: {result.feedback}")
                        break

                parts.append("")

            parts.append("Please generate a better filter that addresses these issues.")
            parts.append("")

        parts.append("Generate the jq filter:")

        return "\n".join(parts)

    def _call_api_with_retry(self, prompt: str) -> str:
        """
        Make the API request with retry logic.

        Args:
            prompt: The user prompt to send.

        Returns:
            The response content from the API.

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails after retries.
            GenerationError: If the response format is invalid.
        """
        last_error: httpx.RequestError | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return self.provider.generate(prompt)

            except httpx.ConnectError as e:
                last_error = e
                error_msg = str(e)

                # Provide helpful error messages for common connection issues
                if (
                    "nodename nor servname provided" in error_msg
                    or "Name or service not known" in error_msg
                ):
                    logger.warning(
                        "DNS resolution failed (attempt %d/%d). "
                        "Verify the endpoint URL or set LLM_BASE_URL environment variable.",
                        attempt + 1,
                        self.MAX_RETRIES,
                    )
                else:
                    logger.warning(
                        "Connection failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        error_msg,
                    )

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_SEC)
                continue

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    "Request error (attempt %d/%d): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    e,
                )

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_SEC)
                continue

        # All retries exhausted
        if isinstance(last_error, httpx.ConnectError):
            error_msg = str(last_error)
            if (
                "nodename nor servname provided" in error_msg
                or "Name or service not known" in error_msg
            ):
                raise GenerationError(
                    "DNS resolution failed. Please verify the endpoint URL is correct "
                    "or set LLM_BASE_URL environment variable to the correct endpoint."
                ) from last_error
            else:
                raise GenerationError(
                    f"Connection failed after {self.MAX_RETRIES} attempts: {error_msg}"
                ) from last_error

        raise GenerationError(
            f"API request failed after {self.MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def _extract(self, response: str) -> str:
        """
        Extract and clean the jq filter from the API response.

        Handles various formatting issues:
        - Removes markdown code blocks
        - Removes 'jq ' prefix
        - Strips outer quotes
        - Takes only code-like lines before comments

        Args:
            response: The raw response from the API.

        Returns:
            The cleaned jq filter expression.
        """
        text = response.strip()

        # Remove markdown code blocks (```jq ... ``` or ``` ... ```)
        code_block_pattern = r"```(?:jq|json)?\s*\n?(.*?)\n?```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # Split into lines and process
        lines = text.split("\n")
        code_lines: list[str] = []

        # Patterns for intro lines to skip (when they're on their own line)
        skip_prefixes = (
            "here is the filter:",
            "here is the jq filter:",
            "the filter is:",
            "the jq filter is:",
            "filter:",
            "jq filter:",
        )

        # Patterns that indicate explanatory text (stop processing)
        explanation_starters = ("this ", "the ")

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Stop at comment lines (lines starting with #)
            if line.startswith("#"):
                break

            line_lower = line.lower()

            # Stop at lines that look like explanations (unless they're our known filter prefixes)
            is_filter_prefix = any(line_lower.startswith(p) for p in skip_prefixes)
            is_explanation = any(line_lower.startswith(p) for p in explanation_starters)

            if is_explanation and not is_filter_prefix:
                # Line starts with "This " or "The " but not a known filter prefix
                break

            # Skip lines that are ONLY intro text (no filter content)
            is_intro_only = False
            for prefix in skip_prefixes:
                if line_lower.startswith(prefix):
                    remainder = line[len(prefix) :].strip()
                    if not remainder:
                        is_intro_only = True
                        break

            if is_intro_only:
                continue

            code_lines.append(line)

        # Take the first code-like line
        if code_lines:
            text = code_lines[0]
        else:
            text = response.strip()

        # Remove common introductory phrases (case insensitive)
        text_lower = text.lower()
        prefixes_to_remove = [
            "here is the filter:",
            "here is the jq filter:",
            "the filter is:",
            "the jq filter is:",
            "filter:",
            "jq filter:",
            "jq ",
        ]

        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix) :].strip()
                break  # Only remove one prefix

        # Strip outer quotes (both single and double)
        if len(text) >= 2:
            if (text.startswith('"') and text.endswith('"')) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1]

        return text.strip()
