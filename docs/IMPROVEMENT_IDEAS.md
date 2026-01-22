# 30 идей для улучшения JQ-Synth

## Список всех идей (краткое описание)

### UX/CLI улучшения (1-10)
1. Добавить live progress indicator с эмодзи и процентами во время synthesis
2. Улучшить error messages с actionable инструкциями и цветным форматированием
3. Добавить --list-tasks для просмотра всех доступных задач в красивом формате
4. Добавить --explain флаг для получения пошагового объяснения сгенерированного фильтра
5. Реализовать fuzzy search для task names (например, "nested" находит "nested-field")
6. Добавить --dry-run режим для оценки стоимости без реального выполнения
7. Добавить интерактивный --setup wizard для первой конфигурации
8. Реализовать --history для просмотра предыдущих успешных решений
9. Добавить --validate-task для проверки корректности task файла перед запуском
10. Добавить colorized output с разными цветами для ошибок/успехов/предупреждений

### Производительность и оптимизация (11-15)
11. Кэширование LLM ответов для идентичных промптов (экономия API вызовов)
12. Параллельное выполнение независимых задач при --task all
13. Streaming API responses для faster feedback (где поддерживается)
14. Предварительная валидация JSON перед API вызовом (fail fast)
15. Batch API requests для multiple tasks (снижение latency)

### Качество кода и архитектура (16-20)
16. Добавить plugin system для кастомных reviewers/scorers
17. Рефакторинг provider abstraction для easier добавления новых LLM
18. Добавить telemetry/metrics collection (opt-in) для улучшения качества
19. Извлечь prompt templates в отдельные файлы для easier tuning
20. Добавить configuration file (~/.jq-by-example/config.yaml) вместо только env vars

### Расширение функциональности (21-25)
21. Добавить --learn режим, который сохраняет успешные паттерны для few-shot learning
22. Реализовать multi-step synthesis для сложных трансформаций (decomposition)
23. Добавить --diff режим для сравнения двух фильтров
24. Добавить export в разные форматы (shell script, jq file, Python snippet)
25. Реализовать --suggest для автоматических рекомендаций по улучшению задачи

### DevEx и тестирование (26-28)
26. Добавить smoke test command (jq-by-example --test) для проверки setup
27. Добавить benchmark suite для regression testing на качестве генерации
28. Реализовать --mock режим для тестирования без реальных API calls

### Безопасность и надёжность (29-30)
29. Добавить rate limiting и exponential backoff для API requests
30. Реализовать sandboxed execution для generated filters (security)

---

## Критическая оценка каждой идеи

### ✅ ПРОШЛИ ОТБОР (будут детально описаны):

**Идея #1: Live progress indicator**
- Оценка: EXCELLENT - критично для UX, пользователь не знает что происходит
- Effort: LOW - просто добавить print statements в orchestrator
- Impact: HIGH - значительно улучшает perceived performance

**Идея #2: Улучшенные error messages**
- Оценка: EXCELLENT - одна из главных проблем сейчас
- Effort: MEDIUM - требует рефакторинг error handling в cli.py
- Impact: HIGH - снижает friction при первом использовании

**Идея #3: --list-tasks**
- Оценка: EXCELLENT - очевидная недостающая функция
- Effort: LOW - просто парсинг tasks.json и форматирование
- Impact: MEDIUM - улучшает discoverability

**Идея #4: --explain флаг**
- Оценка: GOOD - помогает обучению, но требует LLM вызов или hardcoded logic
- Effort: MEDIUM-HIGH - нужен дополнительный LLM prompt или parser
- Impact: MEDIUM - полезно для новичков, но не критично
- Решение: ОТКЛОНЕНА - добавляет сложность, не критично для core UX

**Идея #5: Fuzzy search для task names**
- Оценка: GOOD - nice to have, но не критично
- Effort: LOW - можно использовать simple edit distance
- Impact: LOW - редкий use case (не так много задач)
- Решение: ОТКЛОНЕНА - low impact для effort

**Идея #6: --dry-run режим**
- Оценка: EXCELLENT - важно для cost-conscious пользователей
- Effort: LOW - просто skip executor.run() и показать estimation
- Impact: MEDIUM - снижает anxiety about cost

**Идея #7: --setup wizard**
- Оценка: EXCELLENT - критично для first-time experience
- Effort: MEDIUM - интерактивный input, валидация, сохранение config
- Impact: HIGH - снижает barrier to entry

**Идея #8: --history**
- Оценка: GOOD - полезно, но требует persistence layer
- Effort: MEDIUM-HIGH - нужна БД или file-based storage
- Impact: MEDIUM - nice to have, но не критично
- Решение: ОТКЛОНЕНА для первой итерации - добавляет complexity

**Идея #9: --validate-task**
- Оценка: EXCELLENT - предотвращает waste of API calls
- Effort: LOW - schema validation + heuristics
- Impact: MEDIUM - улучшает developer experience

**Идея #10: Colorized output**
- Оценка: EXCELLENT - современные CLI должны иметь цвета
- Effort: LOW - использовать библиотеку типа colorama или ANSI codes
- Impact: MEDIUM - улучшает readability и UX

**Идея #11: Кэширование LLM ответов**
- Оценка: GOOD - экономит деньги, но добавляет состояние
- Effort: MEDIUM - hash prompts, file/db storage, expiration logic
- Impact: MEDIUM - полезно для development, меньше для production
- Решение: ОТКЛОНЕНА - adds complexity, marginal benefit

**Идея #12: Параллельное выполнение задач**
- Оценка: EXCELLENT - значительное ускорение для --task all
- Effort: MEDIUM - asyncio или ThreadPoolExecutor
- Impact: HIGH - 5x+ speedup для batch mode

**Идея #13: Streaming API responses**
- Оценка: GOOD - faster perceived performance
- Effort: HIGH - требует refactor provider layer для streaming
- Impact: MEDIUM - marginal improvement в latency
- Решение: ОТКЛОНЕНА - сложность не оправдана для gains

**Идея #14: Предварительная валидация JSON**
- Оценка: EXCELLENT - fail fast, экономит API calls
- Effort: LOW - просто json.loads() перед generation
- Impact: MEDIUM - улучшает UX при ошибках ввода

**Идея #15: Batch API requests**
- Оценка: POOR - OpenAI/Anthropic не поддерживают batch для chat
- Effort: N/A
- Impact: N/A
- Решение: ОТКЛОНЕНА - API не поддерживают

**Идея #16: Plugin system**
- Оценка: POOR - overengineering для текущего scope
- Effort: HIGH
- Impact: LOW - нет demand для plugins сейчас
- Решение: ОТКЛОНЕНА - YAGNI

**Идея #17: Рефакторинг provider abstraction**
- Оценка: GOOD - код уже хорош, но можно улучшить
- Effort: MEDIUM
- Impact: LOW - internal quality, не user-facing
- Решение: ОТКЛОНЕНА - код уже достаточно хорош

**Идея #18: Telemetry collection**
- Оценка: POOR - privacy concerns, мало пользы для small project
- Effort: MEDIUM
- Impact: LOW
- Решение: ОТКЛОНЕНА - не нужно сейчас

**Идея #19: Извлечь prompts в файлы**
- Оценка: GOOD - улучшает maintainability
- Effort: LOW - просто move strings to files
- Impact: LOW - только для maintainers
- Решение: ОТКЛОНЕНА - текущий подход работает нормально

**Идея #20: Configuration file**
- Оценка: EXCELLENT - удобнее чем env vars
- Effort: MEDIUM - YAML parsing, validation, merging с env vars
- Impact: MEDIUM - улучшает DX

**Идея #21: --learn режим**
- Оценка: GOOD - интересно, но сложно реализовать правильно
- Effort: HIGH - требует ML или heuristics
- Impact: MEDIUM - uncertain benefit
- Решение: ОТКЛОНЕНА - слишком experimental

**Идея #22: Multi-step synthesis**
- Оценка: GOOD - полезно для complex tasks, но сложно
- Effort: HIGH - orchestration logic
- Impact: MEDIUM - niche use case
- Решение: ОТКЛОНЕНА - добавляет сложность

**Идея #23: --diff режим**
- Оценка: POOR - очень niche use case
- Effort: MEDIUM
- Impact: LOW
- Решение: ОТКЛОНЕНА - low value

**Идея #24: Export в разные форматы**
- Оценка: EXCELLENT - полезно для integration
- Effort: LOW - просто форматирование строк
- Impact: MEDIUM - улучшает workflow integration

**Идея #25: --suggest для улучшения задач**
- Оценка: GOOD - помогает новичкам, но требует LLM call
- Effort: MEDIUM
- Impact: MEDIUM
- Решение: ОТКЛОНЕНА - nice to have, но не критично

**Идея #26: Smoke test command**
- Оценка: EXCELLENT - критично для debugging setup issues
- Effort: LOW - уже есть scripts/smoke_test.py, нужно интегрировать
- Impact: MEDIUM - помогает troubleshooting

**Идея #27: Benchmark suite**
- Оценка: GOOD - полезно для regression testing
- Effort: MEDIUM
- Impact: LOW - только для developers
- Решение: ОТКЛОНЕНА - тесты уже хороши

**Идея #28: --mock режим**
- Оценка: EXCELLENT - критично для testing и development
- Effort: LOW - просто return fixed responses
- Impact: MEDIUM - улучшает testing experience

**Идея #29: Rate limiting**
- Оценка: GOOD - защита от exceeding quotas
- Effort: MEDIUM - tracking requests, delays
- Impact: LOW - редкая проблема
- Решение: ОТКЛОНЕНА - providers уже имеют rate limits

**Идея #30: Sandboxed execution**
- Оценка: GOOD - уже есть timeout/size limits, дополнительная изоляция marginal
- Effort: HIGH - Docker/VM required
- Impact: LOW - jq уже довольно безопасен
- Решение: ОТКЛОНЕНА - текущая security достаточна

---

## ✅ ФИНАЛЬНЫЙ СПИСОК ДЛЯ ИМПЛЕМЕНТАЦИИ

1. **Live progress indicator** (HIGH impact, LOW effort)
2. **Улучшенные error messages** (HIGH impact, MEDIUM effort)
3. **--list-tasks команда** (MEDIUM impact, LOW effort)
4. **--dry-run режим** (MEDIUM impact, LOW effort)
5. **--setup wizard** (HIGH impact, MEDIUM effort)
6. **--validate-task команда** (MEDIUM impact, LOW effort)
7. **Colorized output** (MEDIUM impact, LOW effort)
8. **Параллельное выполнение задач** (HIGH impact, MEDIUM effort)
9. **Предварительная JSON валидация** (MEDIUM impact, LOW effort)
10. **Configuration file support** (MEDIUM impact, MEDIUM effort)
11. **Export в разные форматы** (MEDIUM impact, LOW effort)
12. **Smoke test command** (MEDIUM impact, LOW effort)
13. **--mock режим** (MEDIUM impact, LOW effort)

Отсортировано по приоритету (impact/effort ratio):
1. Live progress indicator
2. Colorized output
3. --list-tasks
4. Предварительная JSON валидация
5. Улучшенные error messages
6. --validate-task
7. --dry-run
8. Export в разные форматы
9. Smoke test command
10. --mock режим
11. Параллельное выполнение
12. Configuration file
13. --setup wizard

---

## Детальные планы имплементации

### ✅ 1. Live Progress Indicator
**Confidence: 95%**

**Что делать:**
Добавить real-time обновления прогресса во время synthesis loop.

**Код:**

```python
# src/orchestrator.py

def solve(self, task: Task, verbose: bool = False) -> Solution:
    logger.info("Starting solve for task '%s'", task.id)

    history: list[Attempt] = []
    best: Attempt | None = None
    stagnation_counter = 0
    seen_filters: set[str] = set()

    for iteration in range(1, self.max_iterations + 1):
        # NEW: Print progress
        if verbose or sys.stdout.isatty():
            print(f"\rIteration {iteration}/{self.max_iterations} 🤖 Generating filter...",
                  end="", flush=True)

        try:
            filter_code = self.generator.generate(task, list(history) if history else None)
        except Exception as e:
            if verbose or sys.stdout.isatty():
                print(f"\r❌ Iteration {iteration}/{self.max_iterations} - Generation failed")
            # ... rest of error handling

        # NEW: Show testing progress
        if verbose or sys.stdout.isatty():
            print(f"\rIteration {iteration}/{self.max_iterations} ⚙️  Testing filter: {filter_code[:50]}...",
                  end="", flush=True)

        attempt = self.reviewer.review(task, filter_code, iteration)

        # NEW: Show score
        if verbose or sys.stdout.isatty():
            emoji = "✓" if attempt.aggregated_score == 1.0 else "📊"
            print(f"\rIteration {iteration}/{self.max_iterations} {emoji} Score: {attempt.aggregated_score:.2f}")
```

**Почему это хорошо:**
- Пользователь всегда знает что происходит
- Perceived performance улучшается даже если actual время то же
- Не требует изменения архитектуры

**Минусы:**
- Может быть раздражающим в non-interactive mode (CI/CD)
- Решение: проверять `sys.stdout.isatty()` или добавить --quiet флаг

---

### ✅ 2. Colorized Output
**Confidence: 98%**

**Что делать:**
Добавить цвета для ошибок (красный), успехов (зелёный), предупреждений (жёлтый).

**Код:**

```python
# src/colors.py (NEW FILE)
"""
ANSI color codes for terminal output.
"""

import sys


class Colors:
    """ANSI color codes."""

    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable all colors (for non-TTY or --no-color)."""
        cls.RED = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.BLUE = ''
        cls.MAGENTA = ''
        cls.CYAN = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.RESET = ''


def should_use_color() -> bool:
    """Check if colors should be used based on environment."""
    # Check NO_COLOR env var (https://no-color.org/)
    if os.environ.get('NO_COLOR'):
        return False

    # Check if stdout is a TTY
    if not sys.stdout.isatty():
        return False

    return True


# Initialize colors based on environment
if not should_use_color():
    Colors.disable()


def success(text: str) -> str:
    """Format text as success (green)."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def error(text: str) -> str:
    """Format text as error (red)."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def warning(text: str) -> str:
    """Format text as warning (yellow)."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


def info(text: str) -> str:
    """Format text as info (blue)."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"
```

```python
# src/cli.py - обновить вывод

from src.colors import success, error, warning, info, Colors

def _print_solution(solution: Solution, verbose: bool = False) -> None:
    status = success("✓") if solution.success else error("✗")
    print(f"\n{status} Task: {Colors.BOLD}{solution.task_id}{Colors.RESET}")
    print(f"  Filter: {Colors.CYAN}{solution.best_filter}{Colors.RESET}")
    print(f"  Score: {_format_score(solution.best_score)}")
    print(f"  Iterations: {solution.iterations_used}")


def _format_score(score: float) -> str:
    """Format score with color based on value."""
    if score == 1.0:
        return success(f"{score:.3f}")
    elif score >= 0.8:
        return warning(f"{score:.3f}")
    else:
        return error(f"{score:.3f}")
```

**Почему это хорошо:**
- Значительно улучшает readability
- Мгновенно видно успехи vs ошибки
- Следует NO_COLOR стандарту

**Минусы:**
- Может не работать на старых Windows terminals
- Решение: автоматическая детекция TTY и NO_COLOR env var

---

### ✅ 3. --list-tasks команда
**Confidence: 100%**

**Что делать:**
Добавить команду для отображения всех доступных задач в красивом формате.

**Код:**

```python
# src/cli.py

def _list_tasks(tasks: list[Task]) -> None:
    """
    Display all available tasks in a formatted table.

    Args:
        tasks: List of Task objects to display.
    """
    from src.colors import Colors, info, warning

    print(f"\n{Colors.BOLD}Available Tasks ({len(tasks)}){Colors.RESET}\n")

    # Group by difficulty (heuristic based on examples count and description)
    basic = []
    intermediate = []
    advanced = []

    for task in tasks:
        difficulty = _estimate_difficulty(task)
        if difficulty == "basic":
            basic.append(task)
        elif difficulty == "intermediate":
            intermediate.append(task)
        else:
            advanced.append(task)

    def print_group(name: str, tasks_list: list[Task]):
        if not tasks_list:
            return
        print(f"{Colors.BOLD}{name}{Colors.RESET} ({len(tasks_list)}):")
        for task in tasks_list:
            examples_text = f"{len(task.examples)} example" + ("s" if len(task.examples) != 1 else "")
            print(f"  • {Colors.CYAN}{task.id:<20}{Colors.RESET} {task.description[:60]}... ({examples_text})")
        print()

    print_group("Basic", basic)
    print_group("Intermediate", intermediate)
    print_group("Advanced", advanced)

    print(f"{Colors.DIM}Usage: jq-by-example --task <task-id>{Colors.RESET}")
    print(f"{Colors.DIM}Run all: jq-by-example --task all{Colors.RESET}\n")


def _estimate_difficulty(task: Task) -> str:
    """Estimate task difficulty based on heuristics."""
    # Simple heuristics
    desc_lower = task.description.lower()

    if any(word in desc_lower for word in ["group", "aggregate", "reduce", "sum all"]):
        return "advanced"
    elif any(word in desc_lower for word in ["filter", "select", "extract multiple"]):
        return "intermediate"
    else:
        return "basic"


# В main():
parser.add_argument(
    "--list-tasks",
    action="store_true",
    help="List all available tasks and exit",
)

# После парсинга:
if parsed.list_tasks:
    try:
        all_tasks = load_tasks(parsed.tasks_file)
        _list_tasks(all_tasks)
        return 0
    except Exception as e:
        print(error(f"Error loading tasks: {e}"), file=sys.stderr)
        return 1
```

**Почему это хорошо:**
- Критичная недостающая функция
- Улучшает discoverable tasks
- Не требует открывать JSON файл

**Минусы:**
- Нет (это очевидная необходимость)

---

### ✅ 4. Предварительная JSON валидация
**Confidence: 100%**

**Что делать:**
Валидировать JSON перед отправкой в LLM для fail-fast.

**Код:**

```python
# src/cli.py

def _validate_json_string(json_str: str, param_name: str) -> tuple[bool, str, Any]:
    """
    Validate a JSON string and provide helpful error messages.

    Args:
        json_str: The JSON string to validate.
        param_name: Name of the parameter (for error messages).

    Returns:
        Tuple of (is_valid, error_message, parsed_data).
    """
    try:
        data = json.loads(json_str)
        return True, "", data
    except json.JSONDecodeError as e:
        # Build helpful error message
        lines = json_str.split('\n')
        error_line = lines[e.lineno - 1] if e.lineno <= len(lines) else ""

        # Detect common issues
        suggestions = []
        if '{' in json_str and '}' not in json_str:
            suggestions.append("Missing closing brace }")
        if '[' in json_str and ']' not in json_str:
            suggestions.append("Missing closing bracket ]")
        if json_str.count('"') % 2 != 0:
            suggestions.append("Unmatched quote \"")
        if "'" in json_str and '"' not in json_str:
            suggestions.append("Use double quotes \" instead of single quotes '")

        msg = f"""
{error('Invalid JSON')} in --{param_name}

{error('Error:')} {e.msg}
{error('Position:')} Line {e.lineno}, Column {e.colno}

{error_line}
{' ' * (e.colno - 1)}^

Common issues:
"""
        for s in suggestions[:3]:  # Limit to 3 suggestions
            msg += f"  • {s}\n"

        msg += f"""
{info('Example of valid JSON:')}
  jq-by-example -i '{{"x": 1}}' -o '1' -d 'Extract x'

{Colors.DIM}Tip: Use a JSON validator: https://jsonlint.com{Colors.RESET}
"""
        return False, msg, None


# В main():
if is_interactive:
    # Interactive mode
    valid, err_msg, input_data = _validate_json_string(parsed.input, "input")
    if not valid:
        print(err_msg, file=sys.stderr)
        return 1

    valid, err_msg, output_data = _validate_json_string(parsed.output, "output")
    if not valid:
        print(err_msg, file=sys.stderr)
        return 1

    example = Example(input_data=input_data, expected_output=output_data)
    task = Task(id="interactive", description=parsed.desc, examples=[example])
    tasks = [task]
```

**Почему это хорошо:**
- Экономит API calls при опечатках
- Значительно улучшает error messages
- Помогает новичкам

**Минусы:**
- Нет

---

### ✅ 5. Улучшенные error messages
**Confidence: 90%**

**Что делать:**
Улучшить все error messages с actionable инструкциями.

**Код:**

```python
# src/cli.py

def _format_jq_not_found_error() -> str:
    """Format helpful error message when jq binary is not found."""
    return f"""
{error('⚠️  jq binary not found in your PATH')}

jq-by-example requires the jq command-line tool to be installed.

{Colors.BOLD}Quick setup:{Colors.RESET}
  • macOS:    {info('brew install jq')}
  • Ubuntu:   {info('sudo apt-get install jq')}
  • Windows:  {info('choco install jq')}

{Colors.BOLD}After installation:{Colors.RESET}
  1. Verify: {Colors.CYAN}jq --version{Colors.RESET}
  2. Try again: {Colors.CYAN}jq-by-example --help{Colors.RESET}

{Colors.DIM}Need help? https://stedolan.github.io/jq/download/{Colors.RESET}
"""


def _format_api_key_error(provider: str) -> str:
    """Format helpful error message when API key is missing."""
    if provider == "openai":
        return f"""
{error('🔑 API key required')}

jq-by-example uses AI to generate jq filters. You need an OpenAI API key.

{Colors.BOLD}Quick setup:{Colors.RESET}
  1. Sign up: {info('https://platform.openai.com')}
  2. Create key: {info('https://platform.openai.com/api-keys')}
  3. Set environment variable:
     {Colors.CYAN}export OPENAI_API_KEY='sk-...'{Colors.RESET}

{Colors.BOLD}Alternative providers:{Colors.RESET}
  • Anthropic: {Colors.CYAN}jq-by-example --provider anthropic{Colors.RESET}
  • Local (Ollama): {Colors.CYAN}jq-by-example --help-local{Colors.RESET}

{Colors.DIM}First time? Try: jq-by-example --setup{Colors.RESET}
"""
    elif provider == "anthropic":
        return f"""
{error('🔑 API key required')}

{Colors.BOLD}Anthropic API setup:{Colors.RESET}
  1. Sign up: {info('https://console.anthropic.com')}
  2. Create key: {info('https://console.anthropic.com/settings/keys')}
  3. Set environment variable:
     {Colors.CYAN}export ANTHROPIC_API_KEY='sk-ant-...'{Colors.RESET}

{Colors.DIM}Tip: OpenAI is the default provider: jq-by-example --provider openai{Colors.RESET}
"""
    else:
        return f"{error('API key required')} for provider: {provider}"


def _format_task_not_found_error(task_id: str, available_tasks: list[Task]) -> str:
    """Format helpful error when task is not found."""
    from difflib import get_close_matches

    available_ids = [t.id for t in available_tasks]
    close_matches = get_close_matches(task_id, available_ids, n=1, cutoff=0.6)

    msg = f"\n{error('⚠️  Task not found:')} {Colors.BOLD}{task_id}{Colors.RESET}\n"

    if close_matches:
        match = close_matches[0]
        matched_task = next(t for t in available_tasks if t.id == match)
        msg += f"\n{warning('Did you mean:')} {Colors.CYAN}{match}{Colors.RESET}\n"
        msg += f"{Colors.DIM}{matched_task.description}{Colors.RESET}\n"

    msg += f"\n{Colors.BOLD}Available tasks:{Colors.RESET}\n"
    for task in available_tasks[:5]:  # Show first 5
        msg += f"  • {Colors.CYAN}{task.id:<20}{Colors.RESET} {task.description[:50]}\n"

    if len(available_tasks) > 5:
        msg += f"\n{Colors.DIM}... and {len(available_tasks) - 5} more{Colors.RESET}\n"

    msg += f"\n{Colors.DIM}View all: jq-by-example --list-tasks{Colors.RESET}\n"

    return msg


# Использовать в main():

try:
    executor = JQExecutor()
except RuntimeError as e:
    if "jq binary not found" in str(e):
        print(_format_jq_not_found_error(), file=sys.stderr)
    else:
        print(error(f"Error: {e}"), file=sys.stderr)
    return 1

try:
    generator = JQGenerator(...)
except ValueError as e:
    if "API key" in str(e) or "api_key" in str(e).lower():
        provider = parsed.provider or "openai"
        print(_format_api_key_error(provider), file=sys.stderr)
    else:
        print(error(f"Error: {e}"), file=sys.stderr)
    return 1

# Task not found:
tasks = [t for t in all_tasks if t.id == parsed.task]
if not tasks:
    print(_format_task_not_found_error(parsed.task, all_tasks), file=sys.stderr)
    return 1
```

**Почему это хорошо:**
- Критичное улучшение first-time experience
- Снижает support burden
- Следует best practices (actionable errors)

**Минусы:**
- Требует maintenance при изменении error paths
- Решение: unit tests для error scenarios

---

### ✅ 6. --validate-task команда
**Confidence: 85%**

**Что делать:**
Добавить валидацию task файлов перед запуском.

**Код:**

```python
# src/task_validator.py (NEW FILE)
"""
Validation logic for task definitions.
"""

from dataclasses import dataclass
from typing import Any

from src.domain import Task


@dataclass
class ValidationWarning:
    """A validation warning for a task."""
    severity: str  # "info", "warning", "error"
    message: str
    suggestion: str


def validate_task(task: Task) -> list[ValidationWarning]:
    """
    Validate a task and return list of warnings/issues.

    Args:
        task: The task to validate.

    Returns:
        List of validation warnings.
    """
    warnings: list[ValidationWarning] = []

    # Check examples count
    if len(task.examples) < 2:
        warnings.append(ValidationWarning(
            severity="warning",
            message=f"Only {len(task.examples)} example provided (recommended: 3+)",
            suggestion="More examples improve accuracy and help AI understand the pattern"
        ))

    # Check if examples are too similar
    if len(task.examples) >= 2:
        inputs = [str(ex.input_data) for ex in task.examples]
        if len(set(inputs)) == 1:
            warnings.append(ValidationWarning(
                severity="warning",
                message="All examples have identical input",
                suggestion="Provide diverse examples to help AI generalize"
            ))

    # Check for very large inputs/outputs
    for i, example in enumerate(task.examples):
        input_size = len(str(example.input_data))
        output_size = len(str(example.expected_output))

        if input_size > 10000:  # 10KB
            warnings.append(ValidationWarning(
                severity="warning",
                message=f"Example {i+1}: Input is very large ({input_size/1024:.1f} KB)",
                suggestion="Large inputs may cause API timeouts. Consider simplifying."
            ))

        if output_size > 10000:
            warnings.append(ValidationWarning(
                severity="warning",
                message=f"Example {i+1}: Output is very large ({output_size/1024:.1f} KB)",
                suggestion="Large outputs may cause issues. Verify this is expected."
            ))

    # Check description quality
    if len(task.description) < 10:
        warnings.append(ValidationWarning(
            severity="info",
            message="Task description is very short",
            suggestion="More detailed descriptions help AI understand the goal"
        ))

    # Check for advanced jq patterns in description
    advanced_keywords = ["group_by", "reduce", "recurse", "walk", "min_by", "max_by"]
    if any(kw in task.description.lower() for kw in advanced_keywords):
        warnings.append(ValidationWarning(
            severity="info",
            message="Task description mentions advanced jq features",
            suggestion="These patterns may be challenging. Consider --max-iters 15"
        ))

    return warnings


def print_validation_warnings(warnings: list[ValidationWarning]) -> None:
    """Print validation warnings in a formatted way."""
    from src.colors import warning, info, error, Colors

    if not warnings:
        print(success("✓ Task validation passed\n"))
        return

    print(f"\n{warning('⚠️  Task validation warnings:')}\n")

    for i, w in enumerate(warnings, 1):
        if w.severity == "error":
            icon = error("✗")
        elif w.severity == "warning":
            icon = warning("⚠️")
        else:
            icon = info("ℹ️")

        print(f"{icon} {w.message}")
        print(f"   {Colors.DIM}→ {w.suggestion}{Colors.RESET}\n")

    print(f"{Colors.DIM}Continue anyway? [y/N]: {Colors.RESET}", end="")
```

```python
# src/cli.py

parser.add_argument(
    "--validate-task",
    action="store_true",
    help="Validate task file and show warnings before running",
)

# В main, перед запуском tasks:
if parsed.validate_task or (not is_interactive and len(tasks) > 0):
    from src.task_validator import validate_task, print_validation_warnings

    for task in tasks:
        warnings = validate_task(task)
        if warnings and parsed.validate_task:
            print(f"\n{Colors.BOLD}Validating task: {task.id}{Colors.RESET}")
            print_validation_warnings(warnings)

            # Ask for confirmation
            response = input().strip().lower()
            if response != 'y':
                print("Aborted.")
                return 1
```

**Почему это хорошо:**
- Предотвращает waste API calls на плохих задачах
- Обучает пользователей best practices
- Экономит время и деньги

**Минусы:**
- Может быть раздражающим для опытных пользователей
- Решение: только при --validate-task или сделать opt-out

---

Теперь имплементирую ТОП-5 идей с наибольшим impact/effort ratio:
1. Colorized output (EASIEST, HIGH IMPACT)
2. --list-tasks (EASY, MEDIUM-HIGH IMPACT)
3. Предварительная JSON валидация (EASY, MEDIUM IMPACT)
4. Live progress indicator (EASY, HIGH IMPACT)
5. Улучшенные error messages (MEDIUM, HIGH IMPACT)
