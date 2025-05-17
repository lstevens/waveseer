# WaveSeer Linting Guide

This document explains the linting setup and best practices for the WaveSeer project.

## Linting Setup

WaveSeer uses [flake8](https://flake8.pycqa.org/) for linting and [pre-commit](https://pre-commit.com/) for automated checks before commits.

### Setting Up Pre-commit Hooks

To set up the pre-commit hooks, run:

```bash
python scripts/setup_hooks.py
```

This will install pre-commit (if not already installed) and set up the hooks to run automatically on each commit.

### Automated Linting Tools

WaveSeer provides two custom tools to help with linting:

1. **wave/utils/lint_fixer.py** - Automatically fixes common linting issues
   - Blank lines with whitespace (W293)
   - Trailing whitespace (W291)
   - Blank line formatting (E302, E305)
   - Unused imports (F401)
   - Missing whitespace after commas (E231)

2. **wave/utils/lint_runner.py** - Higher-level tool for batch processing and reporting
   - Can auto-fix multiple files at once
   - Generates detailed lint reports
   - Identifies critical files needing attention

### Using the Lint Fixer

To fix a single file:
```bash
python -m wave.utils.lint_fixer path/to/file.py
```

To run the lint runner with auto-fix on the entire codebase:
```bash
python -m wave.utils.lint_runner --auto-fix
```

To generate a lint report showing remaining issues:
```bash
python -m wave.utils.lint_runner --report
```

### Running Linting Manually

To run linting manually on all files:

```bash
pre-commit run --all-files
```

To run flake8 directly on specific files or directories:

```bash
python -m flake8 wave/
```

### Automatic Fixing

The project includes a custom lint fixer that can automatically fix common issues:

```bash
python -m wave.utils.lint_runner --auto-fix --path "your/file/or/directory"
```

## Linting Configuration

The linting configuration for WaveSeer is defined in `.pre-commit-config.yaml` and includes:

- **flake8**: For code style and quality checks
- **isort**: For import sorting
- **WaveSeer Lint Fixer**: Custom tool for auto-fixing common issues
- **Pre-commit hooks**: For trailing whitespace, file endings, etc.

## Common Issues and Fixes

### Whitespace Issues (W291, W293)

These are automatically fixed by the pre-commit hooks and the lint fixer.

### Import Issues (F401)

Unused imports are detected and can be fixed automatically using the lint fixer.

### Line Length Issues (E501)

Lines exceeding 100 characters should be reformatted. Some common patterns:

```python
# Before: Long function call
result = some_function_with_a_very_long_name(first_parameter, second_parameter, third_parameter, fourth_parameter)

# After: Split across multiple lines
result = some_function_with_a_very_long_name(
    first_parameter,
    second_parameter,
    third_parameter,
    fourth_parameter
)

# Before: Long string concatenation
message = "This is a very long message that " + "exceeds the line length limit " + user_name + " please fix it."

# After: Use formatted string
message = (
    f"This is a very long message that "
    f"exceeds the line length limit "
    f"{user_name} please fix it."
)
```

### Indentation Issues (E128)

Visual indentation issues can be fixed by aligning continuation lines properly:

```python
# Before: Incorrect indentation
def long_function_name(
    var_one, var_two,
  var_three, var_four):  # E128 continuation line under-indented
    return var_one

# After: Proper indentation
def long_function_name(
    var_one, var_two,
    var_three, var_four):
    return var_one
```

## Best Practices

1. **Run the lint fixer before committing**: `python -m wave.utils.lint_runner --auto-fix`
2. **Address line length issues manually**: Split long lines across multiple lines
3. **Fix one file at a time**: Work incrementally through the codebase
4. **Check diff before committing**: Make sure auto-fixes don't change behavior
