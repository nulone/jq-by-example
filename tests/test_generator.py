"""
Unit tests for generator prompt building and output extraction.

This module tests the JQGenerator class methods for building prompts and
extracting clean jq filter expressions from LLM responses. All tests use
mocked HTTP responses to avoid real API calls.
"""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.domain import Attempt, ErrorType, Example, ExampleResult, Task
from src.generator import GenerationError, JQGenerator


class TestExtractMarkdownRemoval:
    """Tests for _extract method removing markdown code blocks."""

    def test_removes_jq_markdown_fence(self):
        """Markdown code blocks with jq language tag are stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("```jq\n.foo\n```")
            assert result == ".foo"

    def test_removes_plain_markdown_fence(self):
        """Markdown code blocks without language tag are stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("```\n.bar\n```")
            assert result == ".bar"

    def test_removes_json_markdown_fence(self):
        """Markdown code blocks with json language tag are stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("```json\n.baz\n```")
            assert result == ".baz"

    def test_handles_multiline_code_block(self):
        """Multiline content in code blocks takes first line."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("```jq\n.items[].name\n# comment\n```")
            assert result == ".items[].name"

    def test_preserves_filter_without_markdown(self):
        """Plain filter without markdown is preserved."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".simple")
            assert result == ".simple"

    def test_handles_inline_backticks(self):
        """Inline backticks are not treated as code blocks."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            # When there's no proper code block, it should take the first code-like line
            result = generator._extract("``.foo``")
            assert ".foo" in result or result == "``.foo``"


class TestExtractJqPrefixRemoval:
    """Tests for _extract method removing 'jq ' prefix."""

    def test_removes_lowercase_jq_prefix(self):
        """Lowercase 'jq ' prefix is removed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("jq .foo")
            assert result == ".foo"

    def test_removes_uppercase_jq_prefix(self):
        """Uppercase 'JQ ' prefix is removed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("JQ .bar")
            assert result == ".bar"

    def test_removes_mixed_case_jq_prefix(self):
        """Mixed case 'Jq ' prefix is removed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("Jq .baz")
            assert result == ".baz"

    def test_preserves_jq_in_middle(self):
        """'jq' appearing in middle of filter is preserved."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".jq_field")
            assert result == ".jq_field"

    def test_removes_jq_prefix_after_markdown(self):
        """'jq ' prefix inside markdown block is removed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("```\njq .foo\n```")
            assert result == ".foo"


class TestExtractQuoteRemoval:
    """Tests for _extract method removing outer quotes."""

    def test_removes_double_quotes(self):
        """Outer double quotes are stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract('".[].name"')
            assert result == ".[].name"

    def test_removes_single_quotes(self):
        """Outer single quotes are stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("'.[].name'")
            assert result == ".[].name"

    def test_preserves_inner_quotes(self):
        """Quotes inside the filter are preserved."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract('.name == "Alice"')
            assert result == '.name == "Alice"'

    def test_preserves_mismatched_quotes(self):
        """Mismatched quotes are preserved (not stripped)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("\".foo'")
            # Mismatched quotes should not be stripped
            assert result == "\".foo'"

    def test_removes_quotes_after_jq_prefix(self):
        """Quotes are removed after jq prefix is stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract('jq ".foo"')
            assert result == ".foo"


class TestExtractCodeLineSelection:
    """Tests for _extract method taking only code-like lines."""

    def test_stops_at_hash_comment(self):
        """Lines after # comments are excluded."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".foo\n# this extracts foo field")
            assert result == ".foo"

    def test_stops_at_this_explanation(self):
        """Lines starting with 'This ' are treated as explanations."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".bar\nThis filter extracts bar")
            assert result == ".bar"

    def test_stops_at_the_explanation(self):
        """Lines starting with 'The ' are treated as explanations."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".baz\nThe filter above works by...")
            assert result == ".baz"

    def test_skips_empty_lines(self):
        """Empty lines before the filter are skipped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("\n\n.foo\n")
            assert result == ".foo"

    def test_takes_first_code_line(self):
        """Only the first code-like line is used."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".first\n.second\n.third")
            assert result == ".first"

    def test_handles_whitespace_around_filter(self):
        """Whitespace around filter is stripped."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("   .foo   \n")
            assert result == ".foo"


class TestExtractComplexCases:
    """Tests for _extract method with complex/combined scenarios."""

    def test_markdown_with_jq_prefix_and_quotes(self):
        """Handles markdown containing jq prefix and quotes."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract('```jq\njq ".foo"\n```')
            assert result == ".foo"

    def test_response_with_explanation_before_code(self):
        """Handles response with explanation text before the filter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            # The extract method looks for code blocks first
            response = "Here is the filter:\n```jq\n.users[].name\n```"
            result = generator._extract(response)
            assert result == ".users[].name"

    def test_empty_response(self):
        """Empty response returns empty string."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("")
            assert result == ""

    def test_whitespace_only_response(self):
        """Whitespace-only response returns empty string."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("   \n\n   ")
            assert result == ""

    def test_complex_filter_preserved(self):
        """Complex filter expressions are preserved correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            complex_filter = "[.[] | select(.active == true) | {name: .name, id: .id}]"
            result = generator._extract(complex_filter)
            assert result == complex_filter

    def test_pipe_operators_preserved(self):
        """Pipe operators in filter are preserved."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract(".items | map(.x) | add")
            assert result == ".items | map(.x) | add"


class TestExtractIntroductoryPhraseRemoval:
    """Regression tests for removing introductory phrases from LLM responses."""

    def test_removes_here_is_the_filter(self):
        """Removes 'Here is the filter:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("Here is the filter: .user.name")
            assert result == ".user.name"

    def test_removes_here_is_the_jq_filter(self):
        """Removes 'Here is the jq filter:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("Here is the jq filter: .items[]")
            assert result == ".items[]"

    def test_removes_the_filter_is(self):
        """Removes 'The filter is:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("The filter is: .data.id")
            assert result == ".data.id"

    def test_removes_the_jq_filter_is(self):
        """Removes 'The jq filter is:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("The jq filter is: .x + .y")
            assert result == ".x + .y"

    def test_removes_filter_colon(self):
        """Removes 'Filter:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("Filter: .status")
            assert result == ".status"

    def test_removes_jq_filter_colon(self):
        """Removes 'jq filter:' prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("jq filter: .enabled")
            assert result == ".enabled"

    def test_case_insensitive_removal(self):
        """Phrase removal is case insensitive."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("HERE IS THE FILTER: .name")
            assert result == ".name"

    def test_preserves_filter_content(self):
        """Filter content is not modified during prefix removal."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            # Complex filter with spaces and special chars
            result = generator._extract(
                'Here is the filter: [.[] | select(.type == "active") | .id]'
            )
            assert result == '[.[] | select(.type == "active") | .id]'

    def test_only_removes_one_prefix(self):
        """Only the first matching prefix is removed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            # If filter itself contains "filter:", it should be preserved
            result = generator._extract("Filter: .filter")
            assert result == ".filter"

    def test_multiline_with_prefix_on_separate_line(self):
        """Prefix on separate line is skipped, filter on next line is extracted."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            # When prefix is on its own line, it should be skipped
            result = generator._extract("Here is the filter:\n.user.name")
            assert result == ".user.name"

    def test_multiline_with_short_prefix(self):
        """Short prefix on separate line is handled correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            result = generator._extract("Filter:\n.data[].id")
            assert result == ".data[].id"


class TestBuildPromptExamples:
    """Tests for _build_prompt method including examples."""

    def test_includes_task_description(self):
        """Prompt includes the task description."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Extract the name field",
                examples=[Example(input_data={"name": "Alice"}, expected_output="Alice")],
            )

            prompt = generator._build_prompt(task)

            assert "Extract the name field" in prompt

    def test_includes_all_examples(self):
        """Prompt contains all task examples with numbering."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="multi-example",
                description="Test task",
                examples=[
                    Example(input_data={"x": 1}, expected_output=1),
                    Example(input_data={"x": 2}, expected_output=2),
                    Example(input_data={"x": 3}, expected_output=3),
                ],
            )

            prompt = generator._build_prompt(task)

            assert "Example 1:" in prompt
            assert "Example 2:" in prompt
            assert "Example 3:" in prompt

    def test_includes_input_json(self):
        """Prompt includes JSON-formatted input data."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"key": "value"}, expected_output="value")],
            )

            prompt = generator._build_prompt(task)

            assert '{"key": "value"}' in prompt or '{"key":"value"}' in prompt

    def test_includes_expected_output_json(self):
        """Prompt includes JSON-formatted expected output."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=[1, 2, 3])],
            )

            prompt = generator._build_prompt(task)

            assert "[1, 2, 3]" in prompt or "[1,2,3]" in prompt

    def test_example_labels_format(self):
        """Examples are labeled with Input: and Expected Output:."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            prompt = generator._build_prompt(task)

            assert "Input:" in prompt
            assert "Expected Output:" in prompt

    def test_ends_with_generation_prompt(self):
        """Prompt ends with instruction to generate filter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            prompt = generator._build_prompt(task)

            assert "Generate the jq filter:" in prompt


class TestBuildPromptHistory:
    """Tests for _build_prompt method including history feedback."""

    def _make_attempt(
        self,
        filter_code: str,
        score: float,
        error_type: ErrorType,
        feedback: str,
    ) -> Attempt:
        """Helper to create an Attempt with specified properties."""
        example_result = ExampleResult(
            score=score,
            error_type=error_type,
            feedback=feedback,
            actual_output="actual",
            expected_output="expected",
        )
        return Attempt(
            iteration=1,
            filter_code=filter_code,
            example_results=[example_result],
            aggregated_score=score,
            primary_error=error_type,
        )

    def test_includes_history_section(self):
        """Prompt includes previous attempts section when history provided."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".wrong", 0.0, ErrorType.MISSING_EXTRA, "Wrong output"),
            ]

            prompt = generator._build_prompt(task, history)

            assert "Previous attempts" in prompt

    def test_includes_filter_code_in_history(self):
        """History includes the filter codes that were tried."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".wrong_filter", 0.5, ErrorType.MISSING_EXTRA, "Partial match"),
            ]

            prompt = generator._build_prompt(task, history)

            assert ".wrong_filter" in prompt

    def test_includes_score_in_history(self):
        """History includes the scores of previous attempts."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".test", 0.75, ErrorType.ORDER, "Wrong order"),
            ]

            prompt = generator._build_prompt(task, history)

            assert "0.75" in prompt

    def test_includes_error_type_in_history(self):
        """History includes the error types of previous attempts."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".test", 0.0, ErrorType.SYNTAX, "jq error: parse error"),
            ]

            prompt = generator._build_prompt(task, history)

            assert "syntax" in prompt.lower()

    def test_includes_feedback_in_history(self):
        """History includes feedback from failing examples."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".test", 0.0, ErrorType.SHAPE, "Expected list but got dict"),
            ]

            prompt = generator._build_prompt(task, history)

            assert "Expected list but got dict" in prompt

    def test_limits_history_to_last_three(self):
        """Only the last 3 attempts are included in history."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".first", 0.1, ErrorType.SYNTAX, "Error 1"),
                self._make_attempt(".second", 0.2, ErrorType.SHAPE, "Error 2"),
                self._make_attempt(".third", 0.3, ErrorType.ORDER, "Error 3"),
                self._make_attempt(".fourth", 0.4, ErrorType.MISSING_EXTRA, "Error 4"),
                self._make_attempt(".fifth", 0.5, ErrorType.NONE, "Error 5"),
            ]

            prompt = generator._build_prompt(task, history)

            # First two should not be included
            assert ".first" not in prompt
            assert ".second" not in prompt
            # Last three should be included
            assert ".third" in prompt
            assert ".fourth" in prompt
            assert ".fifth" in prompt

    def test_no_history_section_without_history(self):
        """No previous attempts section when history is None."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )

            prompt = generator._build_prompt(task, history=None)

            assert "Previous attempts" not in prompt

    def test_no_history_section_with_empty_history(self):
        """No previous attempts section when history is empty list."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )

            prompt = generator._build_prompt(task, history=[])

            assert "Previous attempts" not in prompt

    def test_includes_improvement_prompt_with_history(self):
        """Prompt asks for better filter when history is provided."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            history = [
                self._make_attempt(".wrong", 0.5, ErrorType.MISSING_EXTRA, "Missing keys"),
            ]

            prompt = generator._build_prompt(task, history)

            assert "better filter" in prompt.lower() or "address" in prompt.lower()


class TestAPIKeyValidation:
    """Tests for API key validation in constructor."""

    def test_missing_api_key_raises_value_error(self):
        """Constructor fails without API key when env var not set."""
        # Ensure env vars are not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                JQGenerator()

            assert "API key required" in str(exc_info.value)

    def test_accepts_api_key_parameter(self):
        """Constructor accepts api_key parameter."""
        generator = JQGenerator(api_key="my-api-key")
        assert generator.provider.api_key == "my-api-key"

    def test_reads_api_key_from_openai_env(self):
        """Constructor reads API key from OPENAI_API_KEY env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}):
            generator = JQGenerator()
            assert generator.provider.api_key == "env-api-key"

    def test_reads_api_key_from_llm_env(self):
        """Constructor reads API key from LLM_API_KEY env var."""
        with patch.dict(os.environ, {"LLM_API_KEY": "llm-api-key"}):
            generator = JQGenerator()
            assert generator.provider.api_key == "llm-api-key"


class TestGenerateWithMockedAPI:
    """Tests for generate method with mocked HTTP responses."""

    def test_successful_generation(self):
        """Successful API call returns extracted filter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Extract name",
                examples=[Example(input_data={"name": "Alice"}, expected_output="Alice")],
            )

            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": ".name"}}]}
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = generator.generate(task)

            assert result == ".name"

    def test_api_timeout_raises_generation_error(self):
        """API timeout raises GenerationError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
                mock_client_class.return_value = mock_client

                with pytest.raises(GenerationError) as exc_info:
                    generator.generate(task)

                assert "timed out" in str(exc_info.value).lower()

    def test_api_http_error_raises_generation_error(self):
        """API HTTP error raises GenerationError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=mock_response,
            )

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                with pytest.raises(GenerationError) as exc_info:
                    generator.generate(task)

                assert "API error" in str(exc_info.value)

    def test_api_request_error_raises_generation_error(self):
        """API request error (network issue) raises GenerationError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.side_effect = httpx.RequestError("Network unreachable")
                mock_client_class.return_value = mock_client

                with pytest.raises(GenerationError) as exc_info:
                    generator.generate(task)

                assert "request failed" in str(exc_info.value).lower()

    def test_invalid_response_format_raises_generation_error(self):
        """Invalid API response format raises GenerationError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={}, expected_output=None)],
            )

            mock_response = MagicMock()
            mock_response.json.return_value = {"invalid": "format"}  # Missing choices
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                with pytest.raises(GenerationError) as exc_info:
                    generator.generate(task)

                assert "Invalid API response" in str(exc_info.value)

    def test_sends_correct_headers(self):
        """API request includes correct authorization header."""
        generator = JQGenerator(api_key="test-api-key-123")
        task = Task(
            id="test-task",
            description="Test",
            examples=[Example(input_data={}, expected_output=None)],
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ".test"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            generator.generate(task)

            # Check that post was called with correct headers
            call_kwargs = mock_client.post.call_args[1]
            assert "Authorization" in call_kwargs["headers"]
            assert "Bearer test-api-key-123" in call_kwargs["headers"]["Authorization"]

    def test_sends_correct_payload_structure(self):
        """API request includes correct payload structure."""
        generator = JQGenerator(api_key="test-key", model="test-model")
        task = Task(
            id="test-task",
            description="Test description",
            examples=[Example(input_data={"x": 1}, expected_output=1)],
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ".x"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            generator.generate(task)

            call_kwargs = mock_client.post.call_args[1]
            payload = call_kwargs["json"]

            assert payload["model"] == "test-model"
            assert "messages" in payload
            assert len(payload["messages"]) == 2  # system + user
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1]["role"] == "user"
            assert "temperature" in payload
            assert "max_tokens" in payload

    def test_extracts_filter_from_markdown_response(self):
        """Filter is correctly extracted from markdown-wrapped response."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"name": "Bob"}, expected_output="Bob")],
            )

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "```jq\n.name\n```"}}]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = generator.generate(task)

            assert result == ".name"

    def test_generate_with_history(self):
        """Generate includes history in the prompt."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            generator = JQGenerator()
            task = Task(
                id="test-task",
                description="Test",
                examples=[Example(input_data={"x": 1}, expected_output=1)],
            )
            example_result = ExampleResult(
                score=0.5,
                error_type=ErrorType.MISSING_EXTRA,
                feedback="Missing element",
                actual_output=[1],
                expected_output=[1, 2],
            )
            history = [
                Attempt(
                    iteration=1,
                    filter_code=".wrong",
                    example_results=[example_result],
                    aggregated_score=0.5,
                    primary_error=ErrorType.MISSING_EXTRA,
                )
            ]

            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": ".x"}}]}
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.post.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = generator.generate(task, history=history)

                # Verify history was included in prompt
                call_kwargs = mock_client.post.call_args[1]
                user_message = call_kwargs["json"]["messages"][1]["content"]
                assert ".wrong" in user_message
                assert "0.50" in user_message


class TestSystemPrompt:
    """Tests for system prompt content."""

    def test_system_prompt_forbids_markdown(self):
        """System prompt instructs not to use markdown."""
        from src.providers import LLMProvider

        assert "NOT" in LLMProvider.SYSTEM_PROMPT
        assert (
            "markdown" in LLMProvider.SYSTEM_PROMPT.lower()
            or "backtick" in LLMProvider.SYSTEM_PROMPT.lower()
        )

    def test_system_prompt_forbids_jq_prefix(self):
        """System prompt instructs not to prefix with 'jq '."""
        from src.providers import LLMProvider

        assert "jq " in LLMProvider.SYSTEM_PROMPT

    def test_system_prompt_forbids_env_vars(self):
        """System prompt forbids $ENV usage."""
        from src.providers import LLMProvider

        assert "$ENV" in LLMProvider.SYSTEM_PROMPT

    def test_system_prompt_forbids_def_statements(self):
        """System prompt forbids def statements."""
        from src.providers import LLMProvider

        assert "def" in LLMProvider.SYSTEM_PROMPT.lower()

    def test_system_prompt_requests_filter_only(self):
        """System prompt requests only the filter expression."""
        from src.providers import LLMProvider

        assert "ONLY" in LLMProvider.SYSTEM_PROMPT


class TestGeneratorConstants:
    """Tests for generator class constants."""

    def test_max_history_attempts_set(self):
        """Max history attempts constant is defined."""
        assert JQGenerator.MAX_HISTORY_ATTEMPTS >= 1

    def test_max_retries_set(self):
        """Max retries constant is defined."""
        assert JQGenerator.MAX_RETRIES >= 1

    def test_retry_delay_reasonable(self):
        """Retry delay is reasonable."""
        assert JQGenerator.RETRY_DELAY_SEC >= 0
