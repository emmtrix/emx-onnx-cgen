from __future__ import annotations

import os
from pathlib import Path


def assert_golden(actual: str, golden_path: Path) -> None:
    if os.getenv("UPDATE_REFS"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual, encoding="utf-8")
        return
    expected = golden_path.read_text(encoding="utf-8")
    if actual != expected:
        message = (
            "Golden mismatch for {path}.\n"
            "--- expected ---\n{expected}\n"
            "--- actual ---\n{actual}\n"
        ).format(path=golden_path, expected=expected, actual=actual)
        raise AssertionError(message)
