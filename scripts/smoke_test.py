#!/usr/bin/env python3
"""
Smoke test for LLM providers.

Quick manual test to verify that a provider is working correctly.
Does not require jq - just checks if the API responds.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import create_provider


def main() -> None:
    """Run a simple smoke test for the specified provider."""
    parser = argparse.ArgumentParser(
        description="Smoke test for LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test OpenAI (default)
  python scripts/smoke_test.py

  # Test specific provider
  python scripts/smoke_test.py --provider anthropic

  # Test with custom model
  python scripts/smoke_test.py --provider openai --model gpt-4o-mini

  # Test Ollama
  python scripts/smoke_test.py --base-url http://localhost:11434/v1 --model llama3
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider to test (default: openai)",
    )
    parser.add_argument(
        "--model",
        help="Model to use (optional, uses provider default)",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for OpenAI-compatible providers (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Smoke Test: {args.provider.upper()}")
    print("=" * 60)
    print()

    # Create provider
    try:
        print(f"[1/3] Initializing {args.provider} provider...")
        provider = create_provider(
            provider_type=args.provider,
            model=args.model,
            base_url=args.base_url,
        )
        print(f"✓ Provider initialized: {type(provider).__name__}")
        if hasattr(provider, "model"):
            print(f"  Model: {provider.model}")
        if hasattr(provider, "endpoint"):
            print(f"  Endpoint: {provider.endpoint}")
        print()
    except ValueError as e:
        print(f"✗ Failed to initialize provider: {e}")
        print()
        print("Make sure you have set the required environment variables:")
        print("  - For OpenAI: OPENAI_API_KEY or LLM_API_KEY")
        print("  - For Anthropic: ANTHROPIC_API_KEY or LLM_API_KEY")
        sys.exit(1)

    # Test API call
    test_prompt = "Generate a simple jq filter that extracts the 'name' field from a JSON object. Respond with only the filter, nothing else."

    try:
        print("[2/3] Sending test request to API...")
        response = provider.generate(test_prompt)
        print("✓ API responded successfully")
        print()

        print("[3/3] Validating response...")
        if not response or not isinstance(response, str):
            print(f"✗ Invalid response type: {type(response)}")
            sys.exit(1)

        print(f"✓ Response is valid (length: {len(response)} chars)")
        print()

        print("=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response.strip())
        print()

        print("=" * 60)
        print("✓ SMOKE TEST PASSED")
        print("=" * 60)
        print()
        print(f"The {args.provider} provider is working correctly!")

    except Exception as e:
        print(f"✗ API request failed: {e}")
        print()
        print("=" * 60)
        print("✗ SMOKE TEST FAILED")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("  1. Check your API key is valid")
        print("  2. Verify your internet connection")
        print("  3. Check the provider's service status")
        print("  4. Try running with --debug flag (not implemented yet)")
        sys.exit(1)


if __name__ == "__main__":
    main()
