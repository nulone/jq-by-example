"""
Unit tests for LLM providers.

This module tests the provider abstraction layer for OpenAI and Anthropic APIs.
"""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.providers import AnthropicProvider, OpenAIProvider, create_provider


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider initialization."""

    def test_requires_api_key(self):
        """Raises ValueError when no API key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIProvider()
            assert "API key required" in str(exc_info.value)

    def test_accepts_api_key_parameter(self):
        """Accepts API key as parameter."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_reads_from_openai_api_key_env(self):
        """Reads API key from OPENAI_API_KEY environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "env-key"

    def test_reads_from_llm_api_key_env(self):
        """Reads API key from LLM_API_KEY environment variable."""
        with patch.dict(os.environ, {"LLM_API_KEY": "llm-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "llm-key"

    def test_uses_default_model(self):
        """Uses default model when not specified."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.model == OpenAIProvider.DEFAULT_MODEL

    def test_accepts_custom_model(self):
        """Accepts custom model parameter."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
        assert provider.model == "gpt-3.5-turbo"

    def test_reads_model_from_env(self):
        """Reads model from LLM_MODEL environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key", "LLM_MODEL": "custom-model"}):
            provider = OpenAIProvider()
            assert provider.model == "custom-model"

    def test_uses_default_base_url(self):
        """Uses default OpenAI base URL when not specified."""
        provider = OpenAIProvider(api_key="test-key")
        assert "api.openai.com" in provider.base_url

    def test_accepts_custom_base_url(self):
        """Accepts custom base URL parameter."""
        provider = OpenAIProvider(api_key="test-key", base_url="https://custom.api/v1")
        assert provider.base_url == "https://custom.api/v1"

    def test_adds_v1_suffix_to_base_url(self):
        """Adds /v1 suffix to base URL if missing."""
        provider = OpenAIProvider(api_key="test-key", base_url="https://custom.api")
        assert provider.base_url.endswith("/v1")

    def test_reads_base_url_from_env(self):
        """Reads base URL from LLM_BASE_URL environment variable."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "key", "LLM_BASE_URL": "https://env.api/v1"}
        ):
            provider = OpenAIProvider()
            assert provider.base_url == "https://env.api/v1"


class TestAnthropicProviderInit:
    """Tests for AnthropicProvider initialization."""

    def test_requires_api_key(self):
        """Raises ValueError when no API key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                AnthropicProvider()
            assert "API key required" in str(exc_info.value)

    def test_accepts_api_key_parameter(self):
        """Accepts API key as parameter."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_reads_from_anthropic_api_key_env(self):
        """Reads API key from ANTHROPIC_API_KEY environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "env-key"

    def test_reads_from_llm_api_key_env(self):
        """Reads API key from LLM_API_KEY environment variable."""
        with patch.dict(os.environ, {"LLM_API_KEY": "llm-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "llm-key"

    def test_uses_default_model(self):
        """Uses default model when not specified."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.model == AnthropicProvider.DEFAULT_MODEL

    def test_accepts_custom_model(self):
        """Accepts custom model parameter."""
        provider = AnthropicProvider(api_key="test-key", model="claude-3-opus")
        assert provider.model == "claude-3-opus"

    def test_reads_model_from_env(self):
        """Reads model from LLM_MODEL environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key", "LLM_MODEL": "custom-model"}):
            provider = AnthropicProvider()
            assert provider.model == "custom-model"


class TestOpenAIProviderGenerate:
    """Tests for OpenAIProvider.generate method."""

    def test_successful_generation(self):
        """Successful API call returns content."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ".foo"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = provider.generate("test prompt")

        assert result == ".foo"

    def test_sends_correct_headers(self):
        """Sends correct authorization headers."""
        provider = OpenAIProvider(api_key="test-api-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ".test"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider.generate("test")

            call_kwargs = mock_client.post.call_args[1]
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"


class TestAnthropicProviderGenerate:
    """Tests for AnthropicProvider.generate method."""

    def test_successful_generation(self):
        """Successful API call returns content."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": ".bar"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = provider.generate("test prompt")

        assert result == ".bar"

    def test_sends_correct_headers(self):
        """Sends correct API key header."""
        provider = AnthropicProvider(api_key="test-api-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": ".test"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider.generate("test")

            call_kwargs = mock_client.post.call_args[1]
            assert "x-api-key" in call_kwargs["headers"]
            assert call_kwargs["headers"]["x-api-key"] == "test-api-key"


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_creates_openai_provider_by_default(self):
        """Creates OpenAI provider when no type specified."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = create_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_creates_openai_provider_explicitly(self):
        """Creates OpenAI provider when type is 'openai'."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = create_provider(provider_type="openai")
            assert isinstance(provider, OpenAIProvider)

    def test_creates_anthropic_provider(self):
        """Creates Anthropic provider when type is 'anthropic'."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_provider(provider_type="anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_reads_provider_type_from_env(self):
        """Reads provider type from LLM_PROVIDER environment variable."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "key"}):
            provider = create_provider()
            assert isinstance(provider, AnthropicProvider)

    def test_raises_on_invalid_provider_type(self):
        """Raises ValueError for invalid provider type."""
        with pytest.raises(ValueError) as exc_info:
            create_provider(provider_type="invalid", api_key="key")
        assert "Invalid provider type" in str(exc_info.value)

    def test_passes_parameters_to_provider(self):
        """Passes parameters to the created provider."""
        provider = create_provider(
            provider_type="openai",
            api_key="test-key",
            model="custom-model",
            base_url="https://custom.api/v1",
        )
        assert provider.api_key == "test-key"
        assert provider.model == "custom-model"
        assert provider.base_url == "https://custom.api/v1"


class TestOpenAIProviderErrorHandling:
    """Tests for OpenAI provider HTTP error handling."""

    def test_handles_http_error_with_message(self):
        """Extracts error message from HTTP error response."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")

    def test_handles_http_error_with_string_error(self):
        """Handles error response with string error field."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")

    def test_handles_http_error_without_json(self):
        """Handles HTTP error when response is not JSON."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")


class TestOpenAIProviderInvalidResponse:
    """Tests for OpenAI provider invalid response handling."""

    def test_raises_on_missing_choices_key(self):
        """Raises RuntimeError when response is missing 'choices' key."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                provider.generate("test")
            assert "Invalid API response format" in str(exc_info.value)


class TestAnthropicProviderInvalidResponse:
    """Tests for Anthropic provider invalid response handling."""

    def test_raises_on_missing_content_key(self):
        """Raises RuntimeError when response is missing 'content' key."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                provider.generate("test")
            assert "Invalid API response format" in str(exc_info.value)


class TestAnthropicProviderErrorHandling:
    """Tests for Anthropic provider HTTP error handling."""

    def test_handles_http_error_with_message(self):
        """Extracts error message from HTTP error response."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")

    def test_handles_http_error_with_string_error(self):
        """Handles error response with string error field."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")

    def test_handles_http_error_without_json(self):
        """Handles HTTP error when response is not JSON."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                provider.generate("test")
