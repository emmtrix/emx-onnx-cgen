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

    # High-signal, operator-agnostic references and workflow hints.
    prompt_lines.append(
        "Helpful references: onnx-org/docs/Operators.md for general operator specs, "
        "onnx-org/onnx/reference/ops/ for numpy reference behavior, "
        "and onnx-org/onnx/backend/test/case/node for backend test inputs."
    )
    prompt_lines.append(
        "Implementation map: add/adjust lowering in src/emx_onnx_cgen/lowering/, "
        "wire codegen in src/emx_onnx_cgen/codegen/c_emitter.py with a matching "
        "templates/*_op.c.j2 file, update runtime/evaluator.py for numpy checks, "
        "and refresh tests/expected_errors entries when support status changes."
    )
    prompt_lines.append(
        "Model inspection hint: use onnx.load(...) and inspect graph.input, "
        "graph.initializer, value_info, and node attributes to understand what is "
        "static vs. dynamic, and what shapes/types are inferred."
    )
    prompt_lines.append(
        "Input loading hint: backend test data only includes non-initializer inputs. "
        "When matching test_data_set_* input_*.pb files to model inputs, filter out "
        "initializers (including sparse initializers) before comparing counts or "
        "assigning data."
    )
    prompt_lines.append(
        "Numerical accuracy hint: if verification fails with small deltas, compare "
        "precision and accumulation order against the ONNX reference implementation "
        "and consider higher-precision accumulators where appropriate."
    )
    prompt_lines.append(
        "CLI hint: use `python -m emx_onnx_cgen ...` (or the emx-onnx-cgen entrypoint) "
        "to run the CLI, since `python -m emx_onnx_cgen.cli` does not invoke main()."
    )
    prompt_lines.append(
        "Debugging hint: use `--keep-temp-dir` (or the printed temp dir path) to "
        "inspect generated C code and compile commands without rerunning the pipeline."
    )
    prompt_lines.append(
        "Codegen wiring hint: new ops usually need updates in name mapping and temp "
        "buffer resolution paths; if codegen crashes on missing attributes, check "
        "the emit-time mapping helpers for that op."
    )
    prompt_lines.append(
        "Artifact hint: pass `--keep-temp-dir` during verification to inspect the "
        "generated C/testbench files when narrowing down failures."
    )
    prompt_lines.append(
        "Shape inference hint: run ONNX shape inference and compare graph inputs "
        "against initializers to confirm which shapes and dtypes are truly dynamic "
        "before adjusting lowering or codegen."
    )
    prompt_lines.append(
        "Pattern hint: locate a similar op in lowering/codegen and mirror its "
        "registration, template wiring, and output-shape handling to avoid "
        "integration gaps."
    )

    prompt_lines.append("\nAnalyze the root cause and implement a fix.")

    # Improved reflection prompt: strictly operator-agnostic, small, actionable.
    prompt_lines.append(
        "At the end, propose up to 3 additions to the 'Helpful references / hints' "
        "section that would have made this fix faster.\n"
        "Constraints:\n"
        "- Must be operator-agnostic (no operator names, attributes, shapes, or special cases).\n"
        "- Must be broadly applicable to many failing tests in this repo.\n"
        "- Prefer actionable workflow guidance (inspection, repro, narrowing root cause, "
        "where to apply changes).\n"
        "- If nothing broadly applicable, write: 'No general additions.'\n"
        "Output format: bullets, each with 'Hint:', 'Why it helps:', 'Where to apply:'."
    )

    print("\n".join(prompt_lines))


if __name__ == "__main__":
    main()
