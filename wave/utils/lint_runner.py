#!/usr/bin/env python3
"""
Linting fix utility for the WaveSeer project.

This script automates the process of fixing common flake8 issues:
1. Whitespace issues (W291, W293) using the lint_fixer module
2. Finding files with the most critical issues for manual fixing
3. Generating reports to track linting progress

Usage:
    python3 -m wave.utils.lint_runner --auto-fix
    python3 -m wave.utils.lint_runner --report
    python3 -m wave.utils.lint_runner --critical-only
"""

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from wave.utils.lint_fixer import process_file


def run_flake8(path: str, select: str = None) -> List[str]:
    """
    Run flake8 on the given path.

    Args:
        path: Path to run flake8 on
        select: Optional comma-separated list of error codes to select

    Returns:
        List of flake8 output lines
    """
    cmd = ["python3", "-m", "flake8", path]
    if select:
        cmd.extend(["--select", select])

    try:
        result = subprocess.run(cmd,
                               capture_output=True,
                               text=True,
                               check=False)
        return result.stdout.splitlines()
    except subprocess.SubprocessError:
        return []


def parse_flake8_output(lines: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Parse flake8 output into a structured format.

    Args:
        lines: List of flake8 output lines

    Returns:
        Dictionary mapping files to error codes and counts
    """
    results = defaultdict(lambda: defaultdict(int))

    for line in lines:
        if not line.strip() or ":" not in line:
            continue

        parts = line.split(":", 3)
        if len(parts) < 3:
            continue

        file_path = parts[0]
        error_code = parts[2].strip().split()[0]

        results[file_path][error_code] += 1

    return results


def auto_fix_whitespace_issues(paths: List[str]) -> Tuple[int, List[str]]:
    """
    Automatically fix whitespace issues in the given paths.

    Args:
        paths: List of file paths to fix

    Returns:
        Tuple of (number of files fixed, list of messages)
    """
    fixed_count = 0
    messages = []

    for file_path in paths:
        try:
            changes_made, file_messages = process_file(Path(file_path))
            messages.extend(file_messages)
            if changes_made:
                fixed_count += 1
        except Exception as e:
            messages.append(f"Error fixing {file_path}: {str(e)}")

    return fixed_count, messages


def find_critical_files(flake8_results: Dict[str, Dict[str, int]],
                     critical_codes: Set[str] = None) -> List[Tuple[str, int, Dict[str, int]]]:
    """
    Find files with critical issues based on error codes.

    Args:
        flake8_results: Results from parse_flake8_output
        critical_codes: Set of error codes considered critical

    Returns:
        List of (file_path, total_errors, error_counts) tuples, sorted by total errors
    """
    if critical_codes is None:
        # Default critical codes: syntax errors, undefined names, assertion errors
        critical_codes = {"E9", "F63", "F7", "F82"}

    critical_files = []

    for file_path, errors in flake8_results.items():
        critical_errors = {code: count for code, count in errors.items()
                          if code[:3] in critical_codes or code in critical_codes}

        total_critical = sum(critical_errors.values())
        if total_critical > 0:
            critical_files.append((file_path, total_critical, critical_errors))

    return sorted(critical_files, key=lambda x: x[1], reverse=True)


def generate_lint_report(flake8_results: Dict[str, Dict[str, int]]) -> str:
    """
    Generate a detailed lint report.

    Args:
        flake8_results: Results from parse_flake8_output

    Returns:
        Formatted report string
    """
    total_errors = sum(sum(counts.values()) for counts in flake8_results.values())
    error_types = defaultdict(int)

    for errors in flake8_results.values():
        for code, count in errors.items():
            error_types[code] += count

    lines = ["# Lint Report", ""]
    lines.append(f"Total files with issues: {len(flake8_results)}")
    lines.append(f"Total errors: {total_errors}")
    lines.append("")

    lines.append("## Error Types")
    for code, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {code}: {count}")
    lines.append("")

    lines.append("## Top Files By Error Count")
    file_counts = [(file, sum(counts.values()))
                  for file, counts in flake8_results.items()]
    for file, count in sorted(file_counts, key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"- {file}: {count} errors")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WaveSeer Lint Runner")
    parser.add_argument("--auto-fix", action="store_true",
                      help="Automatically fix whitespace issues")
    parser.add_argument("--report", action="store_true",
                      help="Generate a detailed lint report")
    parser.add_argument("--critical-only", action="store_true",
                      help="Show only critical issues")
    parser.add_argument("--path", default="wave",
                      help="Path to check (default: wave)")

    args = parser.parse_args()

    print(f"Running flake8 on {args.path}...")
    flake8_output = run_flake8(args.path)
    flake8_results = parse_flake8_output(flake8_output)

    if args.auto_fix:
        print("Automatically fixing whitespace issues...")
        fixed_count, messages = auto_fix_whitespace_issues(flake8_results.keys())
        print(f"Fixed issues in {fixed_count} files.")
        for message in messages:
            print(f"- {message}")

        # Run flake8 again to report remaining issues
        print("\nRemaining issues after auto-fix:")
        flake8_output = run_flake8(args.path)
        flake8_results = parse_flake8_output(flake8_output)

    if args.critical_only:
        critical_files = find_critical_files(flake8_results)
        if critical_files:
            print("\nFiles with critical issues:")
            for file_path, total, errors in critical_files:
                print(f"- {file_path}: {total} critical errors")
                for code, count in errors.items():
                    print(f"  - {code}: {count}")
        else:
            print("\nNo critical issues found!")

    if args.report:
        report = generate_lint_report(flake8_results)
        report_path = Path("lint_report.md")
        report_path.write_text(report)
        print(f"\nGenerated lint report: {report_path}")


if __name__ == "__main__":
    main()
