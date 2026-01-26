#!/usr/bin/env python3
"""Select a random failing test expectation and emit a fix prompt."""

from __future__ import annotations

import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from tests.test_official_onnx_files import (  # noqa: E402
    EXPECTED_ERRORS_ROOT,
    _expected_errors_path_for_repo_relative,
    _list_expectation_repo_paths,
    _load_expectation_for_repo_relative,
)


def load_failing_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    repo_paths = _list_expectation_repo_paths(
        EXPECTED_ERRORS_ROOT,
        path_filter=lambda _: True,
    )
    for repo_relative in repo_paths:
        expectation = _load_expectation_for_repo_relative(repo_relative)
        if (
            expectation.error.startswith("OK")
            or expectation.error == ""
            or "Unsupported elem_type" in expectation.error
        ):
            continue
        json_path = _expected_errors_path_for_repo_relative(repo_relative)
        reproduction_cmd = ""
        if expectation.command_line:
            reproduction_cmd = (
                f"PYTHONPATH=src python -m emx_onnx_cgen {expectation.command_line}"
            )
        entries.append(
            {
                "json_path": str(json_path),
                "error": expectation.error,
                "command_line": reproduction_cmd,
                "operators": ", ".join(expectation.operators or []),
            }
        )
    return entries


def main() -> None:
    entries = load_failing_entries()
    if not entries:
        raise SystemExit("No failing tests found in tests/expected_errors.")

    selection = random.choice(entries)
    prompt_lines = [
        "Please fix the following test failure.",
        "",
        f"JSON file: {selection['json_path']}",
        f"Error message: {selection['error']}",
    ]
    if selection["operators"]:
        prompt_lines.append(f"Operator(s): {selection['operators']}")
    if selection["command_line"]:
        prompt_lines.append(f"Reproduction: {selection['command_line']}")
    prompt_lines.append(
        "Helpful references: onnx-org/docs/Operators.md for operator specs, "
        "onnx-org/onnx/reference/ops/op_<op>.py for numpy reference behavior, "
        "and onnx-org/onnx/backend/test/case/node for test inputs."
    )
    prompt_lines.append(
        "Local ops hint: for com.microsoft operators or local test cases, "
        "check onnx2c-org/test/local_ops for the model generator and "
        "onnx2c-org/src/nodes for a reference implementation."
    )
    prompt_lines.append(
        "Implementation map: add/adjust lowering in src/emx_onnx_cgen/lowering/, "
        "wire codegen in src/emx_onnx_cgen/codegen/c_emitter.py with a matching "
        "templates/*_op.c.j2 file, update runtime/evaluator.py for numpy checks, "
        "and refresh tests/expected_errors entries when support status changes."
    )
    prompt_lines.append(
        "Model inspection hint: when an input is dynamic (e.g., scalar tensors like "
        "TopK's k), check the model's input/output shapes via onnx.load(...) to "
        "see if the value can be inferred from value_info or output shapes."
    )
    prompt_lines.append(
        "Input loading hint: ONNX graphs list initializers in graph.input, but "
        "backend test data only includes non-initializer inputs. When matching "
        "test_data_set_* input_*.pb files to model inputs, filter out initializers "
        "(including sparse initializers) before comparing counts or assigning data."
    )
    prompt_lines.append(
        "Operator behavior hint: consult the ONNX reference op implementation to "
        "capture tie-break rules, optional input defaults, and output ordering so "
        "codegen/runtime match the backend tests."
    )
    prompt_lines.append(
        "Implementation hint: when adding support for a new attribute (like "
        "dilations or ceil_mode), check sibling ops (e.g., MaxPool) for shared "
        "shape formulas and padding logic to keep behavior consistent."
    )
    prompt_lines.append(
        "Codegen hint: if an op lowers successfully but codegen later fails with "
        "missing attributes, check helper methods like _op_output_shape/_op_output_dtype "
        "for missing RotaryEmbedding-style cases."
    )
    prompt_lines.append(
        "CLI hint: use `python -m emx_onnx_cgen ...` (or the emx-onnx-cgen entrypoint) "
        "to run the CLI, since `python -m emx_onnx_cgen.cli` does not invoke main()."
    )
    prompt_lines.append("\nAnalyze the root cause and implement a fix.")
    prompt_lines.append(
        "At the end, reflect on what general information would have helped you fix "
        "the issue more efficiently, and update this script to include that "
        "information in future prompts."
    )

    print("\n".join(prompt_lines))


if __name__ == "__main__":
    main()
