"""Every supported operator has at least one golden C file.

SUPPORT_OPS.md lists an operator as supported once some ONNX file with it
verifies. This module makes sure that the generated C of each of them is also
pinned by a golden file, so that codegen changes show up in review:

* Operators covered by a hand-written case of ``OP_GOLDEN_CASES``
  (``tests/test_golden_ops.py``) need nothing else.
* Every other supported operator has an entry in
  ``tests/golden/operators/manifest.json``: one verified ONNX file from the test
  corpus, compiled into ``tests/golden/operators/<operator>.c``.

``UPDATE_REFS=1 pytest -q tests/test_golden_operators.py`` rewrites the
manifest and the golden files. For each operator it picks the expectation with
the fewest operators, then the smallest model file. The test data
(``onnx-backend-test-data/``, ``emx-ort-test-artifacts-org/``) must be present.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import onnx
import pytest

from emx_onnx_cgen import Compiler
from emx_onnx_cgen.compiler import CompilerOptions
from emx_onnx_cgen.sequence_shape_hints import parse_sequence_element_shape_hints
from golden_utils import assert_golden
from test_golden_ops import OP_GOLDEN_CASES

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_OPS_PATH = REPO_ROOT / "SUPPORT_OPS.md"
EXPECTATIONS_DIR = REPO_ROOT / "tests" / "expected_errors"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "operators"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
REGENERATE_HINT = (
    "Run `UPDATE_REFS=1 pytest -q tests/test_golden_operators.py` to pick a model "
    "for each operator and write the golden files."
)

# Options of a recorded `verify` command line that do not affect the generated
# code, with the number of values each takes.
_VERIFY_ONLY_OPTIONS = {
    "--model-base-dir": 1,
    "--test-data-dir": 1,
    "--test-data-inputs-only": 0,
    "--atol-eps": 1,
    "--max-ulp": 1,
    "--runtime": 1,
}
# Codegen options a golden case may carry. Expectations with any other option
# are not picked.
_CODEGEN_OPTIONS = {"--sequence-element-shape": "sequence_element_shapes"}


def supported_operators() -> set[str]:
    text = SUPPORT_OPS_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"^\| (\S+) \| ✅ \|", text, flags=re.MULTILINE))


def model_operators(model: onnx.ModelProto) -> set[str]:
    """Operators in the format of SUPPORT_OPS.md (top-level graph, domain prefix)."""
    return {
        f"{node.domain}.{node.op_type}" if node.domain else node.op_type
        for node in model.graph.node
    }


def hand_written_operators() -> set[str]:
    operators: set[str] = set()
    for _, _, model_fn in OP_GOLDEN_CASES:
        operators |= model_operators(model_fn())
    return operators


def golden_path(operator: str) -> Path:
    return GOLDEN_DIR / f"{operator}.c"


@dataclass(frozen=True)
class Candidate:
    """A verified ONNX file that can serve as an operator's golden case."""

    model: str
    operator_count: int
    codegen_options: dict[str, list[str]]

    def sort_key(self) -> tuple:
        size = (REPO_ROOT / self.model).stat().st_size
        return (self.operator_count, len(self.codegen_options), size, self.model)

    def manifest_entry(self) -> dict:
        return {"model": self.model, **self.codegen_options}


def _parse_command_line(command_line: str) -> tuple[str, dict[str, list[str]]] | None:
    """The model path and codegen options of a `verify` command line.

    None if the command line uses an option a golden case cannot reproduce.
    """
    tokens = shlex.split(command_line)
    if tokens[:1] != ["verify"]:
        return None
    base_dir, model, codegen = "", None, {}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in _VERIFY_ONLY_OPTIONS:
            if token == "--model-base-dir":
                base_dir = tokens[index + 1]
            index += 1 + _VERIFY_ONLY_OPTIONS[token]
        elif token in _CODEGEN_OPTIONS:
            codegen.setdefault(_CODEGEN_OPTIONS[token], []).append(tokens[index + 1])
            index += 2
        elif token.startswith("-") or model is not None:
            return None
        else:
            model = token
            index += 1
    if model is None:
        return None
    return (Path(base_dir) / model).as_posix(), codegen


def candidates_by_operator() -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = {}
    for path in sorted(EXPECTATIONS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("error", "").startswith("OK") or not data.get("operators"):
            continue
        parsed = _parse_command_line(data.get("command_line", ""))
        if parsed is None:
            continue
        model, codegen = parsed
        if not (REPO_ROOT / model).is_file():
            continue
        candidate = Candidate(model, len(data["operators"]), codegen)
        for operator in data["operators"]:
            candidates.setdefault(operator, []).append(candidate)
    return candidates


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["operators"]


def _write_manifest(entries: dict[str, dict]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "description": (
            "Golden C cases for the supported operators that tests/test_golden_ops.py "
            "does not cover. Generated by tests/test_golden_operators.py with "
            "UPDATE_REFS=1; do not edit."
        ),
        "operators": dict(sorted(entries.items())),
    }
    MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def compile_golden_case(entry: dict, model: onnx.ModelProto | None = None) -> str:
    if model is None:
        model = onnx.load(str(REPO_ROOT / entry["model"]))
    hints = entry.get("sequence_element_shapes")
    options = CompilerOptions(
        sequence_element_shapes=(
            parse_sequence_element_shape_hints(hints) if hints else None
        )
    )
    return Compiler(options).compile(model)


def _regenerate(required: set[str]) -> None:
    candidates = candidates_by_operator()
    unavailable = sorted(op for op in required if op not in candidates)
    if unavailable:
        pytest.fail(
            "No verified ONNX file with a reproducible command line is available "
            f"for {unavailable}. Make sure the test data is present."
        )
    entries = {
        operator: min(candidates[operator], key=Candidate.sort_key).manifest_entry()
        for operator in required
    }
    _write_manifest(entries)
    for path in GOLDEN_DIR.glob("*.c"):
        if path.stem not in entries:
            path.unlink()
    for operator, entry in entries.items():
        assert_golden(compile_golden_case(entry), golden_path(operator))


def test_every_supported_operator_has_a_golden_c_file() -> None:
    supported = supported_operators()
    assert supported, f"No supported operators found in {SUPPORT_OPS_PATH}"
    required = supported - hand_written_operators()
    if os.getenv("UPDATE_REFS"):
        _regenerate(required)

    manifest = load_manifest()
    missing = sorted(required - set(manifest))
    stale = sorted(set(manifest) - required)
    assert not missing, (
        f"Supported operators without a golden C file: {missing}. {REGENERATE_HINT}"
    )
    assert not stale, (
        "tests/golden/operators/manifest.json lists operators that are covered "
        f"by test_golden_ops.py or no longer supported: {stale}. {REGENERATE_HINT}"
    )
    files = {path.stem for path in GOLDEN_DIR.glob("*.c")}
    assert files == set(manifest), (
        f"Golden files without a manifest entry: {sorted(files - set(manifest))}; "
        f"manifest entries without a golden file: {sorted(set(manifest) - files)}. "
        f"{REGENERATE_HINT}"
    )
    # One golden file per operator must stay unique on case-insensitive file systems.
    assert len({name.lower() for name in manifest}) == len(manifest)


@pytest.mark.parametrize("operator", sorted(load_manifest()))
def test_operator_golden(operator: str) -> None:
    entry = load_manifest()[operator]
    if not (REPO_ROOT / entry["model"]).is_file():
        pytest.skip(f"Test data not available: {entry['model']}")
    model = onnx.load(str(REPO_ROOT / entry["model"]))
    assert operator in model_operators(model), (
        f"{entry['model']} no longer contains {operator}. {REGENERATE_HINT}"
    )
    assert_golden(compile_golden_case(entry, model), golden_path(operator))
