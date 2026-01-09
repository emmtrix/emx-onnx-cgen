from __future__ import annotations

from pathlib import Path


def assert_golden(actual: str, golden_path: Path) -> None:
    expected = golden_path.read_text(encoding="utf-8")
    if actual != expected:
        message = (
            "Golden mismatch for {path}.\n"
            "--- expected ---\n{expected}\n"
            "--- actual ---\n{actual}\n"
        ).format(path=golden_path, expected=expected, actual=actual)
        raise AssertionError(message)
