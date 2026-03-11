from __future__ import annotations

from _pytest.terminal import TerminalReporter

from .config import DEFAULT_TESTS, selected_tests_from_env


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config,
) -> None:
    selected = selected_tests_from_env()
    terminalreporter.section("emx-onnx-cgen backend summary", sep="-")
    if selected is None:
        terminalreporter.write_line("Selected tests: all collected backend tests")
    else:
        terminalreporter.write_line(
            "Selected tests: " + ", ".join(selected or DEFAULT_TESTS)
        )
    for key in ("passed", "failed", "skipped"):
        terminalreporter.write_line(
            f"{key}: {len(terminalreporter.stats.get(key, []))}"
        )
