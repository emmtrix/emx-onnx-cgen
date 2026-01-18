#!/usr/bin/env python3
"""Run pytest with duration reporting and write a speed analysis report."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DURATION_LINE_RE = re.compile(
    r"^\s*(?P<seconds>[0-9]+\.?[0-9]*)s\s+(?P<phase>\w+)\s+(?P<nodeid>.+)$"
)
SUMMARY_RE = re.compile(r"=+\s*(?P<summary>.+?)\s+in\s+(?P<seconds>[0-9.]+)s\s*=+")


@dataclass(frozen=True)
class DurationEntry:
    seconds: float
    phase: str
    nodeid: str


@dataclass(frozen=True)
class PytestReport:
    command: list[str]
    exit_code: int
    total_seconds: float | None
    summary: str | None
    durations: list[DurationEntry]
    stdout: str
    stderr: str


def parse_durations(lines: Iterable[str]) -> list[DurationEntry]:
    durations: list[DurationEntry] = []
    in_section = False
    for line in lines:
        if "slowest durations" in line:
            in_section = True
            continue
        if in_section and line.strip().startswith("="):
            in_section = False
            continue
        if not in_section:
            continue
        match = DURATION_LINE_RE.match(line)
        if not match:
            continue
        durations.append(
            DurationEntry(
                seconds=float(match.group("seconds")),
                phase=match.group("phase"),
                nodeid=match.group("nodeid").strip(),
            )
        )
    return durations


def parse_summary(lines: Iterable[str]) -> tuple[str | None, float | None]:
    for line in lines:
        match = SUMMARY_RE.search(line)
        if match:
            return match.group("summary"), float(match.group("seconds"))
    return None, None


def build_command(pytest_args: list[str]) -> list[str]:
    base_args = [
        sys.executable,
        "-m",
        "pytest",
        "--durations=0",
        "--durations-min=0",
    ]
    return base_args + pytest_args


def run_pytest(pytest_args: list[str]) -> PytestReport:
    command = build_command(pytest_args)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_lines = result.stdout.splitlines()
    stderr_lines = result.stderr.splitlines()
    durations = parse_durations(stdout_lines + stderr_lines)
    summary, total_seconds = parse_summary(stdout_lines + stderr_lines)
    return PytestReport(
        command=command,
        exit_code=result.returncode,
        total_seconds=total_seconds,
        summary=summary,
        durations=durations,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def render_report(report: PytestReport, top_n: int) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = [
        "# Pytest Speed Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Run Summary",
        f"- Command: `{' '.join(report.command)}`",
        f"- Exit code: {report.exit_code}",
    ]
    if report.summary is not None:
        lines.append(f"- Summary: {report.summary}")
    if report.total_seconds is not None:
        lines.append(f"- Total duration: {report.total_seconds:.3f}s")

    lines.extend(["", "## Slowest Durations", ""])

    if not report.durations:
        lines.append("No duration data was captured from pytest output.")
    else:
        sorted_durations = sorted(report.durations, key=lambda entry: entry.seconds, reverse=True)
        top_entries = sorted_durations[:top_n]
        lines.append("| Rank | Duration (s) | Phase | Test |")
        lines.append("| ---: | ---: | --- | --- |")
        for index, entry in enumerate(top_entries, start=1):
            lines.append(
                f"| {index} | {entry.seconds:.3f} | {entry.phase} | `{entry.nodeid}` |"
            )
        lines.append("")
        lines.append(
            f"Captured {len(report.durations)} duration entries (showing top {len(top_entries)})."
        )

    return "\n".join(lines) + "\n"


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest with duration reporting and write a speed analysis report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pytest_speed_report.md"),
        help="Path to write the report file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of slowest tests to include in the report.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to pytest (prefix with -- to separate).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    report = run_pytest(pytest_args)
    content = render_report(report, args.top)
    write_report(args.output, content)

    if report.exit_code != 0:
        print("Pytest exited with a non-zero status. Report generated.", file=sys.stderr)

    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
