# JQ-By-Example

**AI-Powered JQ Filter Synthesis Tool**

JQ-By-Example automatically generates [jq](https://stedolan.github.io/jq/) filter expressions from input/output JSON examples using LLM-powered synthesis with iterative refinement.

[![CI](https://github.com/nulone/jq-by-example/actions/workflows/ci.yml/badge.svg)](https://github.com/nulone/jq-by-example/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](demo.gif)

## Overview

JQ-By-Example solves a common developer problem: you know what JSON transformation you want, but writing the correct jq filter is tricky. Simply provide example input/output pairs, and JQ-By-Example will synthesize the filter for you.

**Key Features:**

- 🤖 **LLM-Powered Generation** - Uses OpenAI, Anthropic, or compatible APIs to generate filter candidates
- 🔄 **Iterative Refinement** - Automatically improves filters based on algorithmic feedback
- ✅ **Verified Correctness** - Executes filters against real jq binary to verify outputs
- 📊 **Detailed Diagnostics** - Classifies errors (syntax, shape, missing keys, order) with partial scoring
- 🛡️ **Safe Execution** - Sandboxed jq execution with timeout and output limits
- 🔒 **Production-Ready** - Comprehensive edge case handling, security auditing, structured logging

## Installation

### Prerequisites

1. **Python 3.10 or higher**
2. **jq binary** installed and available in PATH:
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   sudo apt-get install jq

   # Windows (with chocolatey)
   choco install jq
   ```

### Install JQ-By-Example

```bash
git clone https://github.com/nulone/jq-by-example.git
cd jq-by-example
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Quick Start

### Interactive Mode

Synthesize a filter from a single input/output example:

```bash
jq-by-example \
  --input '{"user": {"name": "Alice", "age": 30}}' \
  --output '"Alice"' \
  --desc "Extract the user's name"
```

Output:
```
============================================================
[1/1] Solving: interactive
Description: Extract the user's name
Examples: 1
Max iterations: 10
============================================================

✓ Task: interactive
  Filter: .user.name
  Score: 1.000
  Iterations: 1
  Time: 2.34s

============================================================
OVERALL SUMMARY
============================================================
Tasks: 1/1 passed (100.0%)
Total time: 2.34s
Average time per task: 2.34s
============================================================
```

### Batch Mode

Run predefined tasks from a file:

```bash
# Run a specific task
jq-by-example --task nested-field

# Run all tasks
jq-by-example --task all

# With verbose output (shows iteration details)
jq-by-example --task all --verbose
```

## CLI Options

```
usage: jq-by-example [-h] [-t TASK] [--tasks-file TASKS_FILE] [--max-iters MAX_ITERS]
                [--baseline] [-i INPUT] [-o OUTPUT] [-d DESC]
                [--provider {openai,anthropic}] [--model MODEL] [--base-url BASE_URL]
                [-v] [--debug]

AI-Powered JQ Filter Synthesis Tool

options:
  -h, --help            Show this help message and exit

Task Selection:
  -t TASK, --task TASK  Task ID to run, or 'all' to run all tasks
  --tasks-file TASKS_FILE
                        Path to tasks JSON file (default: data/tasks.json)

Iteration Control:
  --max-iters MAX_ITERS
                        Maximum iterations per task (default: 10)
  --baseline            Single-shot mode (max_iterations=1, no refinement)

Interactive Mode:
  -i INPUT, --input INPUT
                        Input JSON for interactive mode
  -o OUTPUT, --output OUTPUT
                        Expected output JSON for interactive mode
  -d DESC, --desc DESC  Task description for interactive mode

LLM Provider:
  --provider {openai,anthropic}
                        LLM provider type (default: from LLM_PROVIDER env or 'openai')
  --model MODEL         Model identifier (default: from LLM_MODEL env or provider default)
  --base-url BASE_URL   Base URL for OpenAI-compatible providers (default: from LLM_BASE_URL env)

Output Control:
  -v, --verbose         Enable verbose output (shows iteration details)
  --debug               Enable debug logging (shows detailed internal state)
```

### Usage Examples

```bash
# Interactive mode - simple field extraction
jq-by-example -i '{"x": 42}' -o '42' -d 'Extract x'

# Interactive mode - array filtering
jq-by-example -i '[1,2,3,4,5]' -o '[2,4]' -d 'Keep only even numbers'

# Interactive mode - nested object access
jq-by-example \
  -i '{"data": {"users": [{"name": "Alice"}]}}' \
  -o '["Alice"]' \
  -d 'Extract all user names'

# Batch mode - run specific task
jq-by-example --task nested-field

# Batch mode - all tasks with verbose output
jq-by-example --task all --verbose

# Single-shot mode (no refinement) for baseline comparison
jq-by-example --task nested-field --baseline

# Custom tasks file
jq-by-example --task my-task --tasks-file my-tasks.json

# Debug mode for troubleshooting
jq-by-example --task nested-field --debug

# Limit iterations
jq-by-example --task filter-active --max-iters 5

# Use Anthropic provider
jq-by-example --provider anthropic --task nested-field

# Use specific model
jq-by-example --model gpt-4o-mini --task nested-field

# Use OpenRouter
jq-by-example --base-url https://openrouter.ai/api/v1 --model anthropic/claude-3.5-sonnet --task nested-field

# Use local Ollama
jq-by-example --base-url http://localhost:11434/v1 --model llama3 --task nested-field
```

## How It Works

JQ-By-Example uses a **deterministic oracle** approach:

1. **Generation**: An LLM (GPT-4, Claude, or compatible model) generates candidate jq filters based on your examples and description
2. **Verification**: Each filter is executed against the real jq binary with your input examples
3. **Scoring**: A deterministic algorithm compares actual vs expected outputs, computing similarity scores (0.0 to 1.0)
4. **Feedback**: The algorithm classifies errors (syntax, shape, missing/extra elements, order) and generates actionable feedback
5. **Refinement**: The LLM receives the feedback and generates an improved filter
6. **Iteration**: Steps 2-5 repeat until a perfect match is found or limits are reached

This hybrid approach combines LLM creativity with deterministic verification, ensuring correctness while leveraging AI for filter synthesis.

## Architecture

JQ-By-Example follows a modular architecture with clear separation of concerns:

```
┌──────────┐
│   CLI    │  Entry point, argument parsing, output formatting
└────┬─────┘
     │
     ▼
┌────────────────┐
│  Orchestrator  │  Manages synthesis loop, tracks progress
└─┬──────────┬───┘
  │          │
  ▼          ▼
┌──────────┐ ┌──────────┐
│Generator │ │ Reviewer │  Filter evaluation & scoring
│(LLM)     │ └────┬─────┘
└──────────┘      │
                  ▼
               ┌──────────┐
               │ Executor │  Sandboxed jq execution
               └──────────┘
```

### Components

#### 1. CLI (`src/cli.py`)
- Parses command-line arguments
- Loads tasks from JSON files
- Formats and displays results with progress indicators
- Tracks timing and generates summaries

#### 2. Orchestrator (`src/orchestrator.py`)
- Manages the iterative refinement loop
- Coordinates between Generator and Reviewer
- Implements anti-stuck protocols:
  - Duplicate filter detection (normalized)
  - Stagnation detection (no improvement for N iterations)
  - Max iteration limit
- Tracks best solution and complete history

#### 3. Generator (`src/generator.py`)
- Interfaces with LLM providers (OpenAI, Anthropic, or compatible APIs)
- Builds prompts with task description, examples, and feedback history
- Extracts clean filter code from LLM responses
- Implements retry logic with exponential backoff
- Includes security features (API key never logged, input truncation)

#### 4. Reviewer (`src/reviewer.py`)
- Evaluates generated filters against examples
- Computes similarity scores using:
  - Jaccard similarity for lists
  - Key/value matching for objects
  - Exact matching for scalars
- Classifies errors by priority (SYNTAX → SHAPE → MISSING_EXTRA → ORDER)
- Generates actionable feedback for refinement

#### 5. Executor (`src/executor.py`)
- Safely executes jq binary in subprocess
- Enforces resource limits (timeout, output size)
- Prevents shell injection (uses argument list, not shell)
- Handles jq errors and timeouts gracefully

#### 6. Domain (`src/domain.py`)
- Defines core data structures (Task, Example, Attempt, Solution)
- Uses frozen dataclasses for immutability
- Type-safe with full type hints

### Data Flow

1. **User** provides task (JSON examples + description) via CLI
2. **CLI** loads/validates task, initializes components
3. **Orchestrator** starts synthesis loop:
   - Iteration 1: Calls **Generator** with task only
   - **Generator** queries LLM API for filter candidate
   - **Reviewer** evaluates filter using **Executor**
   - **Executor** runs jq binary with filter on examples
   - **Reviewer** computes scores and generates feedback
   - Iteration 2+: **Generator** receives history/feedback
   - Loop continues until perfect match or limits reached
4. **Orchestrator** returns **Solution** with best filter, score, history
5. **CLI** displays formatted results with timing information

### Error Classification

The reviewer classifies errors by priority (highest to lowest):

| Error Type | Description | Example | Score |
|------------|-------------|---------|-------|
| `SYNTAX` | Invalid jq filter syntax | `invalid[[[` | 0.0 |
| `SHAPE` | Wrong output type | Expected `[]`, got `{}` | 0.0 |
| `MISSING_EXTRA` | Missing or extra elements/keys | Expected `[1,2,3]`, got `[1,2]` | 0.67 (Jaccard) |
| `ORDER` | Correct elements, wrong order | Expected `[1,2,3]`, got `[3,2,1]` | 0.8 |
| `NONE` | Perfect match | - | 1.0 |

### Scoring Algorithm

- **Lists**: Jaccard similarity = `|intersection| / |union|`
  - Special case: Correct elements, wrong order = 0.8
- **Dicts**: `(key_similarity + value_match_ratio) / 2`
- **Scalars**: Binary (1.0 for exact match, 0.0 for mismatch)
- **Multiple examples**: Arithmetic mean of scores

## Supported jq Patterns

JQ-By-Example works well with these common jq operations:

- **Field extraction**: `.foo`, `.user.name`, `.data.items[0]`
- **Array operations**: `.[]`, `.[0]`, `.[1:3]`, `.[-1]`
- **Filtering**: `select(.active == true)`, `select(.age > 18)`
- **Mapping**: `map(.name)`, `[.[] | .id]`
- **Array construction**: `[.items[].name]`
- **Object construction**: `{name: .user.name, email: .user.email}`
- **Conditionals**: `if .status == "active" then .name else null end`
- **Null handling**: `select(. != null)`, `.field // "default"`
- **String operations**: String interpolation, concatenation
- **Arithmetic**: Addition, subtraction, comparison operators
- **Type checking**: `type`, `length`

## Known Limitations

JQ-By-Example may struggle with these advanced jq features:

- **Aggregations**: `group_by()`, `reduce`, `min_by()`, `max_by()`
- **Complex recursion**: `recurse()`, `walk()`
- **Variable bindings**: Complex `as $var` patterns
- **Custom functions**: `def` statements (blocked for security)
- **Advanced array operations**: `combinations()`, `transpose()`
- **Path manipulation**: `getpath()`, `setpath()`, `delpaths()`
- **Format strings**: `@csv`, `@json`, `@base64`

For these cases, you may need to write the filter manually or break down the task into simpler steps.

## Model recommendations

| Task complexity | Recommended model | Speed |
|-----------------|-------------------|-------|
| Simple filters (extract, select) | GPT-4o-mini, Claude Haiku | Fast |
| Medium (grouping, aggregation, recursion) | Claude Sonnet, GPT-4o | Fast |
| Complex algorithms (graph traversal, sorting) | DeepSeek R1 | Slow (minutes) |

> Note: DeepSeek R1 solved topological sort and Dijkstra's shortest path in jq. Most users won't need this — standard models handle 95%+ of real-world tasks.

## Supported Providers

| Provider | Status | Note |
|----------|--------|------|
| OpenAI | Stable ✅ | Default provider |
| Anthropic | Beta ⚠️ | Different API format |
| OpenRouter | Tested ✅ | OpenAI-compatible |
| Ollama | Alpha 🧪 | Local only, requires setup |

> Note: OpenAI is default and most tested. Others should work but report issues if found.

### Provider Setup

**OpenAI (Default)**

```bash
export OPENAI_API_KEY='sk-...'
# Optional: specify model (default: gpt-4o)
export LLM_MODEL='gpt-4o'
```

**Anthropic**

```bash
export LLM_PROVIDER='anthropic'
export ANTHROPIC_API_KEY='sk-ant-...'
# Optional: specify model (default: claude-sonnet-4-20250514)
export LLM_MODEL='claude-sonnet-4-20250514'
```

**OpenRouter**

```bash
export LLM_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_API_KEY='sk-or-...'
export LLM_MODEL='anthropic/claude-3.5-sonnet'
```

**Local (Ollama)**

```bash
export LLM_BASE_URL='http://localhost:11434/v1'
export LLM_MODEL='llama3'
export OPENAI_API_KEY='dummy'  # Ollama doesn't require a real key
```

**Together AI / Groq**

```bash
# Together AI
export LLM_BASE_URL='https://api.together.xyz/v1'
export OPENAI_API_KEY='...'

# Groq
export LLM_BASE_URL='https://api.groq.com/openai/v1'
export OPENAI_API_KEY='gsk_...'
```

## Task File Format

Tasks are defined in JSON format:

```json
{
  "tasks": [
    {
      "id": "nested-field",
      "description": "Extract the user's name from a nested object structure",
      "examples": [
        {
          "input": {"user": {"name": "Alice", "age": 30}},
          "expected_output": "Alice"
        },
        {
          "input": {"user": {"name": "Bob", "email": "bob@example.com"}},
          "expected_output": "Bob"
        }
      ]
    }
  ]
}
```

### Guidelines for Good Tasks

1. **Provide 3+ examples** for better generalization
2. **Include edge cases**: empty arrays, null values, missing fields
3. **Be specific** in descriptions: "Extract user names" vs "Transform data"
4. **Use diverse inputs**: different structures help the LLM understand the pattern
5. **Test edge cases**: null, empty arrays/objects, deeply nested (3+ levels), special characters in keys

### Built-in Tasks

The `data/tasks.json` file includes these example tasks:

| Task ID | Description | Difficulty | Expected Filter |
|---------|-------------|------------|-----------------|
| `nested-field` | Extract `.user.name` | Easy | `.user.name` |
| `filter-active` | Filter where `active == true` | Medium | `[.[] \| select(.active == true)]` |
| `extract-emails` | Extract emails, skip null/missing | Medium | `[.[].email \| select(. != null)]` |

## Troubleshooting

### "jq binary not found"

**Problem**: JQ-By-Example can't locate the jq executable.

**Solution**: Ensure jq is installed and in your PATH:

```bash
# Check if jq is installed
which jq

# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# Verify installation
jq --version
```

### "API key required"

**Problem**: Missing API key environment variable.

**Solution**: Set the appropriate API key for your provider:

```bash
# For OpenAI
export OPENAI_API_KEY='sk-...'

# For Anthropic
export ANTHROPIC_API_KEY='sk-ant-...'

# Or use generic variable
export LLM_API_KEY='...'

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### "API request failed: DNS resolution failed"

**Problem**: DNS resolution failed for the API endpoint.

**Solution**:
1. Check your internet connection
2. Verify the API endpoint is correct:
   ```bash
   # For OpenAI
   curl -I https://api.openai.com/v1/chat/completions

   # For Anthropic
   curl -I https://api.anthropic.com/v1/messages
   ```
3. If using a custom endpoint, check `LLM_BASE_URL`:
   ```bash
   export LLM_BASE_URL='https://api.openai.com/v1'
   ```

### "API request timed out"

**Problem**: API request has a 60-second timeout. Connection issues or server problems.

**Solution**:
- Check your internet connection
- Try again (transient network issues)
- Check your provider's service status
- Reduce task complexity (fewer examples, simpler description)

### "Connection failed after 3 attempts"

**Problem**: Multiple retry attempts failed.

**Solution**:
1. Verify API endpoint is reachable:
   ```bash
   # For OpenAI
   curl https://api.openai.com/v1/chat/completions

   # For custom endpoint
   curl $LLM_BASE_URL/chat/completions
   ```
2. Check your firewall/proxy settings
3. Try with `--debug` flag to see detailed error messages

### Filter works in jq but not in JQ-By-Example

**Problem**: Your filter works when you run it manually with jq, but fails in JQ-By-Example.

**Cause**: JQ-By-Example uses these jq flags: `-M` (monochrome) and `-c` (compact output).

**Solution**: Ensure your expected output matches compact JSON format:
```bash
# Wrong: pretty-printed JSON
{
  "name": "Alice"
}

# Correct: compact JSON
{"name":"Alice"}
```

### Low success rate or poor quality filters

**Problem**: Filters don't match expected outputs, or require many iterations.

**Solution**:
1. **Improve task description**: Be specific about what transformation you want
2. **Add more examples**: 3+ examples help the LLM generalize better
3. **Include edge cases**: Empty arrays, null values, missing keys
4. **Simplify the task**: Break complex transformations into smaller tasks
5. **Use verbose mode**: `--verbose` to see iteration details and understand failures

### Debug mode for troubleshooting

Enable debug logging to see detailed internal state:

```bash
jq-by-example --task my-task --debug
```

Debug mode shows:
- Full API request/response details (with truncation for security)
- Detailed scoring calculations
- Duplicate filter detection
- Stagnation counter progression

## Security

JQ-By-Example implements production-ready security measures:

### API Key Protection
- API keys are **never logged** (even in debug mode)
- Stored securely in environment variables
- Transmitted only via HTTPS headers

### Input Sanitization
- Large inputs are **truncated in logs** (max 100 characters)
- Prevents accidental exposure of sensitive data in log files

### Shell Injection Prevention
- jq filters passed as subprocess **arguments** (not via shell)
- No use of `shell=True` in subprocess calls
- Filters are never interpolated into shell commands

### Resource Limits
- Timeout: 1 second per filter execution
- Max output: 1 MB per execution
- Prevents denial-of-service attacks and resource exhaustion

### Edge Case Handling
Comprehensive test coverage for:
- Null input/output
- Empty arrays and objects
- Deeply nested structures (3+ levels)
- Special characters in keys (spaces, unicode, @, -)
- Large arrays (100+ items)
- Type mismatches and conversions

## Development

### Setup Development Environment

```bash
git clone https://github.com/nulone/jq-by-example.git
cd jq-by-example
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run unit tests (no API key required)
pytest -m "not e2e"

# Run all tests including E2E (requires API key)
export OPENAI_API_KEY='your-key-here'
# or
export ANTHROPIC_API_KEY='your-key-here'
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_generator.py -v
```

### Code Quality

```bash
# Type checking
mypy src

# Linting
ruff check src tests

# Formatting
ruff format src tests

# Run all checks (recommended before commit)
ruff check src tests && \
ruff format --check src tests && \
mypy src && \
pytest -m "not e2e"
```

## Project Structure

```
jq-by-example/
├── src/
│   ├── cli.py           # CLI entry point
│   ├── orchestrator.py  # Synthesis loop coordinator
│   ├── generator.py     # LLM-based filter generation
│   ├── providers.py     # LLM provider abstractions (OpenAI, Anthropic)
│   ├── reviewer.py      # Filter evaluation & scoring
│   ├── executor.py      # Safe jq execution
│   ├── domain.py        # Core data structures
│   └── security.py      # Security utilities (log truncation)
├── tests/
│   ├── test_cli.py
│   ├── test_orchestrator.py
│   ├── test_generator.py
│   ├── test_reviewer.py
│   ├── test_executor.py
│   ├── test_domain.py
│   ├── test_edge_cases.py  # Production-ready edge cases
│   └── test_e2e.py         # End-to-end tests (require API key)
├── data/
│   └── tasks.json       # Example task definitions
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Make** your changes with tests
4. **Ensure** all checks pass:
   ```bash
   ruff check src tests
   ruff format --check src tests
   mypy src
   pytest -m "not e2e"
   ```
5. **Commit** with clear messages: `git commit -m "Add feature X"`
6. **Push** to your fork: `git push origin feature/my-feature`
7. **Open** a Pull Request

## Code Style

- Type hints required for all public functions
- Docstrings required for all public functions and classes (Google style)
- 100 character line limit
- Follow existing patterns in codebase
- Add tests for all new features
- Security-first mindset (never log sensitive data)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [jq](https://stedolan.github.io/jq/) - The excellent JSON processor by Stephen Dolan
- [OpenAI](https://openai.com) - GPT models and API
- [Anthropic](https://anthropic.com) - Claude models and API

---

**JQ-By-Example** - Because life's too short to debug jq filters manually.
