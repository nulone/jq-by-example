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
