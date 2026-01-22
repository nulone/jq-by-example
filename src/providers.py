"""
LLM providers for jq filter generation.

This module provides an abstraction layer for different LLM APIs (OpenAI, Anthropic, etc.)
to generate jq filter expressions.
"""

import json
import logging
import os
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    SYSTEM_PROMPT = """You are a jq filter expert. Generate a single jq filter expression that transforms the input JSON to produce the expected output.

Rules:
- Output ONLY the jq filter expression, nothing else
- Do NOT use markdown code blocks or backticks
- Do NOT prefix with 'jq ' command
- Do NOT use $ENV or environment variables
- Do NOT use 'def' statements or custom functions
- The filter must work with standard jq

Respond with just the filter, for example: .users[].name"""

    TEMPERATURE = 0.3
    MAX_TOKENS = 500
    TIMEOUT_SEC = 60.0
    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 1.0

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.

        Returns:
            The response content from the LLM.

        Raises:
            Exception: If the API call fails.
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible API provider.

    Works with OpenAI, OpenRouter, Together, Groq, Ollama, and other
    OpenAI-compatible endpoints.

    Attributes:
        api_key: The API key for authentication.
        model: The model identifier to use.
        base_url: The base URL for the API endpoint.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: API key. If not provided, reads from LLM_API_KEY or OPENAI_API_KEY.
            model: Model identifier. If not provided, reads from LLM_MODEL or uses default.
            base_url: Base URL for API. If not provided, reads from LLM_BASE_URL or uses
                OpenAI default.

        Raises:
            ValueError: If no API key is provided and environment variables are not set.
        """
        # Resolve API key
        resolved_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not resolved_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set LLM_API_KEY or "
                "OPENAI_API_KEY environment variable."
            )

        # SECURITY: API key is stored but never logged
        self.api_key = resolved_key

        # Resolve model
        self.model = model or os.environ.get("LLM_MODEL") or self.DEFAULT_MODEL

        # Resolve base URL
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"

        # Ensure base_url ends with /v1 for compatibility
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"

        self.endpoint = f"{self.base_url}/chat/completions"

        logger.debug(
            "OpenAIProvider initialized with model=%s, endpoint=%s",
            self.model,
            self.endpoint,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response using OpenAI-compatible API.

        Args:
            prompt: The user prompt to send.

        Returns:
            The response content from the API.

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails.
            RuntimeError: If the response format is invalid.
        """
        # SECURITY: API key used in headers but never logged
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
        }

        logger.debug(
            "Calling OpenAI-compatible API with model=%s, endpoint=%s",
            self.model,
            self.endpoint,
        )

        with httpx.Client(timeout=self.TIMEOUT_SEC) as client:
            response = client.post(
                self.endpoint,
                headers=headers,
                json=payload,
            )

            # Handle HTTP errors with proper error message extraction
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    # Extract error message without logging full response
                    if "error" in error_data:
                        if isinstance(error_data["error"], dict):
                            error_msg = error_data["error"].get("message", error_msg)
                        else:
                            error_msg = str(error_data["error"])
                except Exception:
                    pass  # Use default error message if parsing fails

                logger.error("API error: %s", error_msg)
                response.raise_for_status()

        # Parse response
        try:
            data = response.json()
            content: str = data["choices"][0]["message"]["content"]
            logger.debug("API response received (%d chars)", len(content))
            return content

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Invalid API response format: %s", e)
            raise RuntimeError(f"Invalid API response format: {e}") from e


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider.

    Uses the native Anthropic Messages API.

    Attributes:
        api_key: The Anthropic API key.
        model: The model identifier to use.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, reads from LLM_API_KEY or
                ANTHROPIC_API_KEY.
            model: Model identifier. If not provided, reads from LLM_MODEL or uses default.

        Raises:
            ValueError: If no API key is provided and environment variables are not set.
        """
        # Resolve API key
        resolved_key = (
            api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        )

        if not resolved_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set LLM_API_KEY or "
                "ANTHROPIC_API_KEY environment variable."
            )

        # SECURITY: API key is stored but never logged
        self.api_key = resolved_key

        # Resolve model
        self.model = model or os.environ.get("LLM_MODEL") or self.DEFAULT_MODEL

        self.endpoint = "https://api.anthropic.com/v1/messages"

        logger.debug(
            "AnthropicProvider initialized with model=%s, endpoint=%s",
            self.model,
            self.endpoint,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response using Anthropic Messages API.

        Args:
            prompt: The user prompt to send.

        Returns:
            The response content from the API.

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails.
            RuntimeError: If the response format is invalid.
        """
        # SECURITY: API key used in headers but never logged
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "system": self.SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug(
            "Calling Anthropic API with model=%s, endpoint=%s",
            self.model,
            self.endpoint,
        )

        with httpx.Client(timeout=self.TIMEOUT_SEC) as client:
            response = client.post(
                self.endpoint,
                headers=headers,
                json=payload,
            )

            # Handle HTTP errors with proper error message extraction
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    # Extract error message without logging full response
                    if "error" in error_data:
                        if isinstance(error_data["error"], dict):
                            error_msg = error_data["error"].get("message", error_msg)
                        else:
                            error_msg = str(error_data["error"])
                except Exception:
                    pass  # Use default error message if parsing fails

                logger.error("API error: %s", error_msg)
                response.raise_for_status()

        # Parse response
        try:
            data = response.json()
            content: str = data["content"][0]["text"]
            logger.debug("API response received (%d chars)", len(content))
            return content

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Invalid API response format: %s", e)
            raise RuntimeError(f"Invalid API response format: {e}") from e


def create_provider(
    provider_type: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_type: Provider type ('openai' or 'anthropic'). If not provided,
            reads from LLM_PROVIDER environment variable (default: 'openai').
        api_key: API key for the provider.
        model: Model identifier.
        base_url: Base URL (only for OpenAI-compatible providers).

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If provider_type is invalid or required credentials are missing.
    """
    resolved_type = (provider_type or os.environ.get("LLM_PROVIDER") or "openai").lower()

    if resolved_type == "openai":
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    elif resolved_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Invalid provider type: {resolved_type}. Must be 'openai' or 'anthropic'."
        )
