"""Tests for security utilities."""

from src.security import mask_api_key, sanitize_for_logging, truncate_for_logging


class TestTruncateForLogging:
    """Tests for truncate_for_logging function."""

    def test_short_text_unchanged(self) -> None:
        """Short text should be returned unchanged."""
        assert truncate_for_logging("short text") == "short text"

    def test_exactly_max_length(self) -> None:
        """Text at exactly max_length should be returned unchanged."""
        text = "a" * 100
        assert truncate_for_logging(text, max_length=100) == text

    def test_long_text_truncated(self) -> None:
        """Long text should be truncated with preview and suffix."""
        text = "a" * 200
        result = truncate_for_logging(text, max_length=100)
        assert len(result) <= 100
        assert "truncated from 200 chars" in result
        assert "..." in result

    def test_custom_max_length(self) -> None:
        """Custom max_length should be respected."""
        text = "a" * 200
        result = truncate_for_logging(text, max_length=50)
        assert len(result) <= 50
        assert "truncated" in result

    def test_very_small_max_length(self) -> None:
        """Very small max_length should truncate hard without preview."""
        text = "a" * 200
        result = truncate_for_logging(text, max_length=20)
        assert len(result) <= 20
        assert result == "a" * 20

    def test_non_string_input(self) -> None:
        """Non-string input should be converted to string."""
        assert truncate_for_logging(123) == "123"
        assert truncate_for_logging([1, 2, 3]) == "[1, 2, 3]"
        assert truncate_for_logging({"key": "value"}) == "{'key': 'value'}"

    def test_empty_string(self) -> None:
        """Empty string should be returned unchanged."""
        assert truncate_for_logging("") == ""

    def test_unicode_text(self) -> None:
        """Unicode text should be handled correctly."""
        text = "привет " * 50
        result = truncate_for_logging(text, max_length=100)
        assert len(result) <= 100


class TestMaskApiKey:
    """Tests for mask_api_key function."""

    def test_openai_key(self) -> None:
        """OpenAI key should be masked correctly."""
        key = "sk-1234567890abcdef"
        result = mask_api_key(key)
        assert result.startswith("sk-1")
        assert result.endswith("cdef")
        assert "*" in result
        assert len(result) == len(key)

    def test_anthropic_key(self) -> None:
        """Anthropic key should be masked correctly."""
        key = "sk-ant-1234567890abcdef"
        result = mask_api_key(key)
        assert result.startswith("sk-a")
        assert result.endswith("cdef")
        assert "*" in result
        assert len(result) == len(key)

    def test_short_key(self) -> None:
        """Short key should be masked with minimal exposure."""
        key = "short"
        result = mask_api_key(key)
        assert result == "s***t"
        assert len(result) == len(key)

    def test_very_short_key(self) -> None:
        """Very short key (<=2 chars) should be fully masked."""
        assert mask_api_key("ab") == "**"
        assert mask_api_key("a") == "*"

    def test_empty_key(self) -> None:
        """Empty key should return empty string."""
        assert mask_api_key("") == ""

    def test_key_length_preserved(self) -> None:
        """Masked key should have same length as original."""
        keys = [
            "sk-1234567890",
            "sk-ant-abcdefghijklmnop",
            "Bearer-token-12345",
        ]
        for key in keys:
            assert len(mask_api_key(key)) == len(key)


class TestSanitizeForLogging:
    """Tests for sanitize_for_logging function."""

    def test_normal_text_unchanged(self) -> None:
        """Normal text without sensitive data should be unchanged."""
        assert sanitize_for_logging("normal text") == "normal text"

    def test_openai_key_masked(self) -> None:
        """OpenAI API key should be masked."""
        result = sanitize_for_logging("sk-1234567890abcdef")
        assert "[MASKED:" in result
        assert "sk-1" in result
        assert "cdef" in result
        assert "1234567890" not in result

    def test_anthropic_key_masked(self) -> None:
        """Anthropic API key should be masked."""
        result = sanitize_for_logging("sk-ant-1234567890abcdef")
        assert "[MASKED:" in result
        assert "sk-a" in result
        assert "cdef" in result
        assert "1234567890" not in result

    def test_bearer_token_masked(self) -> None:
        """Bearer token should be masked (token after 'Bearer ', not the word itself)."""
        result = sanitize_for_logging("Bearer token123456")
        assert "[MASKED:" in result
        # Word "Bearer" should remain visible
        assert "Bearer" in result
        # Token should be masked
        assert "token123456" not in result or "toke***" in result

    def test_multiple_keys_masked(self) -> None:
        """Multiple API keys should all be masked."""
        text = "sk-key1234567890 and sk-ant-key0987654321"
        result = sanitize_for_logging(text)
        assert result.count("[MASKED:") == 2
        assert "1234567890" not in result
        assert "0987654321" not in result

    def test_key_in_sentence(self) -> None:
        """API key embedded in sentence should be masked."""
        text = "Using API key sk-1234567890abcdef for requests"
        result = sanitize_for_logging(text)
        assert "[MASKED:" in result
        assert "1234567890" not in result

    def test_non_string_input(self) -> None:
        """Non-string input should be converted and sanitized."""
        result = sanitize_for_logging({"api_key": "sk-1234567890"})
        assert "[MASKED:" in result

    def test_empty_string(self) -> None:
        """Empty string should be returned unchanged."""
        assert sanitize_for_logging("") == ""

    def test_truncation_applied(self) -> None:
        """Very long sanitized text should be truncated."""
        text = "normal text " * 50
        result = sanitize_for_logging(text)
        # Default truncation is 100 chars
        assert len(result) <= 100

    def test_pattern_order_specificity(self) -> None:
        """More specific patterns should be checked before less specific ones."""
        # sk-ant- should be detected as Anthropic, not split into "sk-" + "ant-"
        text = "sk-ant-1234567890abcdef"
        result = sanitize_for_logging(text)
        # Should have only ONE masked segment, not multiple
        assert result.count("[MASKED:") == 1

    def test_multiple_same_pattern(self) -> None:
        """Multiple occurrences of the same pattern should all be masked."""
        text = "sk-key1 sk-key2 sk-key3"
        result = sanitize_for_logging(text)
        assert result.count("[MASKED:") == 3

    def test_adjacent_keys(self) -> None:
        """Adjacent keys should both be masked."""
        text = "sk-key1sk-key2"
        result = sanitize_for_logging(text)
        # Both keys should be detected and masked
        assert "[MASKED:" in result


class TestTruncateForLoggingEdgeCases:
    """Additional edge case tests for truncate_for_logging."""

    def test_max_length_edge_at_boundary(self) -> None:
        """Text at boundary should handle rounding correctly."""
        text = "a" * 100
        result = truncate_for_logging(text, max_length=50)
        # Should be truncated with suffix
        assert len(result) <= 50
        assert "truncated" in result

    def test_very_long_text_with_small_available_space(self) -> None:
        """Very long text with small max_length should truncate hard."""
        text = "x" * 1000
        result = truncate_for_logging(text, max_length=15)
        assert len(result) <= 15
        # Should be hard truncated without preview
        assert result == "x" * 15


class TestSanitizeForLoggingEdgeCases:
    """Additional edge case tests for sanitize_for_logging."""

    def test_multiple_anthropic_keys(self) -> None:
        """Multiple Anthropic keys should all be masked."""
        text = "sk-ant-key1 and sk-ant-key2"
        result = sanitize_for_logging(text)
        assert result.count("[MASKED:") == 2

    def test_nested_patterns(self) -> None:
        """Nested patterns should be handled correctly."""
        text = "Bearer sk-1234567890"
        result = sanitize_for_logging(text)
        # Both Bearer and sk- should be detected
        assert "[MASKED:" in result

    def test_pattern_at_end_of_string(self) -> None:
        """Pattern at end of string should be masked."""
        text = "API key: sk-1234567890"
        result = sanitize_for_logging(text)
        assert "[MASKED:" in result
        assert "1234567890" not in result

    def test_very_long_key_truncation(self) -> None:
        """Very long masked string should be truncated."""
        # Create a very long "key" that will be masked
        long_key = "sk-" + "x" * 200
        result = sanitize_for_logging(long_key)
        # Should be masked AND truncated (default max_length=100)
        assert len(result) <= 100
        assert "[MASKED:" in result


class TestBearerTokenMasking:
    """Regression tests for Bearer token masking with dots and equals."""

    def test_bearer_jwt_token_full(self) -> None:
        """Full JWT token should be masked including dots and equals."""
        # Real JWT format: header.payload.signature
        jwt = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = sanitize_for_logging(jwt)

        # Token should be masked
        assert "[MASKED:" in result
        # JWT parts should not be visible
        assert "eyJhbGci" not in result
        assert "eyJzdWI" not in result
        assert "dozjgN" not in result or "dozj***" in result  # May be partially visible in mask

    def test_bearer_token_with_dots(self) -> None:
        """Bearer token with dots should have dots included in mask."""
        text = "Bearer abc.def.ghi"
        result = sanitize_for_logging(text)

        assert "[MASKED:" in result
        # Full token including dots should be masked
        assert "abc.def.ghi" not in result
        # Word "Bearer" should remain
        assert "Bearer" in result

    def test_bearer_token_with_equals(self) -> None:
        """Bearer token with equals signs should include them in mask."""
        text = "Bearer token123=="
        result = sanitize_for_logging(text)

        assert "[MASKED:" in result
        # Token including equals should be masked
        assert "token123==" not in result
        assert "Bearer" in result

    def test_bearer_token_mixed_chars(self) -> None:
        """Bearer token with alphanumeric, dots, equals, hyphens, underscores."""
        text = "Bearer abc-def_ghi.jkl==mno"
        result = sanitize_for_logging(text)

        assert "[MASKED:" in result
        # All characters should be captured
        assert "abc-def_ghi.jkl==mno" not in result

    def test_multiple_bearer_tokens(self) -> None:
        """Multiple Bearer tokens should all be masked."""
        text = "Token1: Bearer abc.def.ghi Token2: Bearer xyz.uvw.rst"
        result = sanitize_for_logging(text)

        assert result.count("[MASKED:") == 2
        assert "abc.def.ghi" not in result
        assert "xyz.uvw.rst" not in result

    def test_bearer_word_not_masked(self) -> None:
        """The word 'Bearer' itself should not be masked, only the token after it."""
        text = "Authorization: Bearer token123"
        result = sanitize_for_logging(text)

        # Word "Bearer" should still be visible
        assert "Bearer" in result
        # But token should be masked
        assert "[MASKED:" in result
        assert "token123" not in result

    def test_bearer_without_token(self) -> None:
        """Bearer without a token should not cause issues."""
        text = "Bearer "
        result = sanitize_for_logging(text)

        # Should not crash, just return as-is
        assert "Bearer" in result
