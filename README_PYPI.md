# JQ-By-Example

[![PyPI](https://img.shields.io/pypi/v/jq-by-example)](https://pypi.org/project/jq-by-example/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered jq filter synthesis from input/output JSON examples.**

You know what JSON transformation you want, but writing the jq filter is tricky? Just provide examples — JQ-By-Example synthesizes the filter for you.

## Installation
```bash
pip install jq-by-example
```

Requires `jq` binary: `brew install jq` (macOS) or `apt install jq` (Linux)

## Quick Start
```bash
export OPENAI_API_KEY='sk-...'

jq-by-example \
  --input '{"user": {"name": "Alice"}}' \
  --output '"Alice"' \
  --desc "Extract name"

# Output: .user.name
```

## Features

- 🤖 **LLM-Powered** — Uses OpenAI, Anthropic, or local Ollama
- 🔄 **Iterative Refinement** — Automatically improves filters based on feedback
- ✅ **Verified** — Executes against real jq binary to ensure correctness
- 📊 **Diagnostics** — Detailed error classification and scoring

## Documentation

Full documentation: **[github.com/nulone/jq-by-example](https://github.com/nulone/jq-by-example)**
