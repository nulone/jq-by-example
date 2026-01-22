"""
Security utilities for safe logging and data handling.

This module provides utilities to prevent sensitive data exposure in logs
and ensure safe handling of user-provided data.

IMPORTANT SECURITY PRACTICES:
- API keys are NEVER logged (stored in environment variables only)
- Prompts are logged as length + hash only, never full content
- HTTP headers are never logged (contain API keys)
- Error responses extract only error messages, not full response bodies
"""

from typing import Any


def truncate_for_logging(data: Any, max_length: int = 100) -> str:
    """
    Truncate data for safe logging to prevent exposure of large sensitive content.

    Args:
        data: The data to truncate (will be converted to string).
        max_length: Maximum length of the returned string. Defaults to 100.

    Returns:
        Truncated string representation of the data.

    Examples:
        >>> truncate_for_logging("short text")
        'short text'
        >>> truncate_for_logging("a" * 200, max_length=50)
        'aaaaa...aaaaa (truncated from 200 chars)'
    """
    data_str = str(data)
    if len(data_str) <= max_length:
        return data_str

    # Calculate space for suffix
    suffix = f" (truncated from {len(data_str)} chars)"
    available = max_length - len(suffix) - 3  # -3 for "..."

    if available < 10:
        # Not enough space for preview, just truncate hard
        return data_str[:max_length]

    # Show beginning and end with ellipsis
    preview_len = available // 2
    result = f"{data_str[:preview_len]}...{data_str[-preview_len:]}{suffix}"

    # Ensure we don't exceed max_length due to rounding
    if len(result) > max_length:
        result = result[:max_length]

    return result


def mask_api_key(api_key: str) -> str:
    """
    Mask an API key for logging purposes.

    Shows only the first 4 and last 4 characters, masking the rest.
    Preserves the original length to aid debugging.

    Args:
        api_key: The API key to mask.

    Returns:
        Masked version of the API key with the same length as the original.

    Examples:
        >>> mask_api_key("sk-1234567890abcdef")
        'sk-1***********cdef'
        >>> mask_api_key("short")
        's***t'
    """
    if len(api_key) <= 8:
        # For very short keys, mask all but first and last char
        if len(api_key) <= 2:
            return "*" * len(api_key)
        return f"{api_key[0]}{'*' * (len(api_key) - 2)}{api_key[-1]}"

    # For normal keys, show first 4 and last 4, preserve length
    middle_len = len(api_key) - 8
    return f"{api_key[:4]}{'*' * middle_len}{api_key[-4:]}"


def sanitize_for_logging(value: Any) -> str:
    """
    Sanitize a value for safe logging.

    This function detects and masks potential sensitive data like API keys,
    tokens, passwords, etc. Handles multiple occurrences of sensitive patterns.

    Args:
        value: The value to sanitize.

    Returns:
        Sanitized string representation.

    Examples:
        >>> sanitize_for_logging("sk-1234567890abcdef")
        '[MASKED:sk-1****cdef]'
        >>> sanitize_for_logging("normal text")
        'normal text'
        >>> sanitize_for_logging("sk-key1 and sk-key2")
        '[MASKED:sk-k****key1] and [MASKED:sk-k****key2]'
    """
    value_str = str(value)

    # Detect potential API keys or tokens
    # IMPORTANT: Order from most specific to least specific!
    sensitive_patterns = [
        "sk-ant-",  # Anthropic keys (more specific, check first)
        "sk-",  # OpenAI keys (less specific)
        "Bearer ",  # Bearer tokens (JWT, etc.)
    ]

    # Use temporary placeholders that won't match any pattern
    # Using a very unique pattern to avoid collisions with user data
    replacements: list[tuple[str, str]] = []
    placeholder_counter = 0

    for pattern in sensitive_patterns:
        # Process ALL occurrences of each pattern
        offset = 0
        while True:
            start_idx = value_str.find(pattern, offset)
            if start_idx == -1:
                break  # No more occurrences

            # Skip if this position is already inside a placeholder
            # Check if start_idx is between any existing replacement position
            is_already_masked = False
            temp_pos = 0
            for i in range(placeholder_counter):
                # Use highly unique placeholder to avoid collision with user data
                placeholder_marker = f"<<<__MASK_SENTINEL_{i}__>>>"
                pos = value_str.find(placeholder_marker, temp_pos)
                if pos != -1 and pos <= start_idx < pos + len(placeholder_marker):
                    is_already_masked = True
                    break

            if is_already_masked:
                offset = start_idx + 1
                continue

            # Extract what looks like a key
            # For Bearer tokens: skip "Bearer " prefix and extract the actual token
            # For API keys: include the prefix (sk-, sk-ant-, etc.)
            is_bearer = pattern == "Bearer "

            if is_bearer:
                # For Bearer: extract token after "Bearer " (not the word itself)
                # JWT format: xxx.yyy.zzz= (contains dots and equals)
                key_start = start_idx + len(pattern)  # Skip "Bearer "
            else:
                # For API keys: extract from pattern start (includes sk-, sk-ant-, etc.)
                key_start = start_idx

            key_end = key_start
            for i in range(key_start, len(value_str)):
                char = value_str[i]
                if char.isalnum() or char in "-_":
                    key_end = i + 1
                elif is_bearer and char in ".=":
                    # JWT tokens contain dots and equals
                    key_end = i + 1
                else:
                    break

            if key_end > key_start:
                key = value_str[key_start:key_end]
                masked = mask_api_key(key)
                # Use highly unique temporary placeholder to avoid collision with user data
                temp_placeholder = f"<<<__MASK_SENTINEL_{placeholder_counter}__>>>"
                final_replacement = f"[MASKED:{masked}]"
                replacements.append((temp_placeholder, final_replacement))
                placeholder_counter += 1

                value_str = value_str[:key_start] + temp_placeholder + value_str[key_end:]
                # Update offset to continue searching after this replacement
                offset = key_start + len(temp_placeholder)
            else:
                # Pattern found but no valid key extracted, skip past it
                offset = start_idx + len(pattern)

    # Replace temporary placeholders with final masked values
    for temp, final in replacements:
        value_str = value_str.replace(temp, final)

    return truncate_for_logging(value_str)
