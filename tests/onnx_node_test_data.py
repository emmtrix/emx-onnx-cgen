"""Materialize the official ONNX node backend tests on disk.

Since onnx 1.23 (onnx/onnx#7959) neither the repository nor the wheel ships
``onnx/backend/test/data/node``; upstream builds the node tests in memory from
``onnx.backend.test.case.node``. The official-file tests, their expectation
files and the ``verify`` command lines still address the node models by their
historical location below the ``onnx-org`` submodule, so this module writes
them back there in the pre-1.23 layout (``model.onnx`` plus
``test_data_set_N/{input,output}_K.pb``). The submodule ignores untracked
content, so the generated files never show up as submodule changes.

Run ``python tests/onnx_node_test_data.py`` to generate the files outside of
pytest, e.g. before reproducing a single ``verify`` command line.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_ORG_ROOT = REPO_ROOT / "onnx-org"
NODE_DATA_ROOT = ONNX_ORG_ROOT / "onnx" / "backend" / "test" / "data" / "node"
VERSION_STAMP_NAME = ".onnx-version"


class NodeTestDataError(RuntimeError):
    """The node test data cannot be generated for the pinned onnx version."""


def _submodule_onnx_version() -> str | None:
    version_path = ONNX_ORG_ROOT / "VERSION_NUMBER"
    if not version_path.exists():
        return None
    return version_path.read_text(encoding="utf-8").strip()


def _stamped_version(root: Path) -> str | None:
    stamp = root / VERSION_STAMP_NAME
    if not stamp.exists():
        return None
    return stamp.read_text(encoding="utf-8").strip()


def _to_proto(value: Any, value_info: Any) -> Any:
    from onnx import TensorProto, numpy_helper

    value_type = value_info.type
    if value_type.HasField("map_type"):
        return numpy_helper.from_dict(value, value_info.name)
    if value_type.HasField("sequence_type"):
        return numpy_helper.from_list(value, value_info.name)
    if value_type.HasField("optional_type"):
        return numpy_helper.from_optional(value, value_info.name)
    if isinstance(value, TensorProto):
        return value
    return numpy_helper.from_array(value, value_info.name)


def _write_case(case: Any, case_dir: Path) -> None:
    case_dir.mkdir(parents=True)
    graph = case.model.graph
    (case_dir / "model.onnx").write_bytes(case.model.SerializeToString())
    for set_index, (inputs, outputs) in enumerate(case.data_sets):
        data_set_dir = case_dir / f"test_data_set_{set_index}"
        data_set_dir.mkdir()
        for index, value in enumerate(inputs):
            proto = _to_proto(value, graph.input[index])
            (data_set_dir / f"input_{index}.pb").write_bytes(proto.SerializeToString())
        for index, value in enumerate(outputs):
            proto = _to_proto(value, graph.output[index])
            (data_set_dir / f"output_{index}.pb").write_bytes(proto.SerializeToString())


def _generate(target: Path, onnx_version: str) -> None:
    from onnx.backend.test.case.node import collect_testcases

    # The upstream case definitions compute some expected outputs that are
    # intentionally inf/nan; their numpy RuntimeWarnings are noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cases = collect_testcases(None)
    for case in cases:
        _write_case(case, target / case.name)
    (target / VERSION_STAMP_NAME).write_text(onnx_version + "\n", encoding="utf-8")
    LOGGER.info("Generated %d ONNX node backend tests in %s", len(cases), target)


def ensure_node_test_data(root: Path = NODE_DATA_ROOT) -> bool:
    """Generate the node tests below ``root`` unless they are up to date.

    Returns ``False`` without touching ``root`` when the ``onnx-org`` submodule
    is not checked out. Raises :class:`NodeTestDataError` when the installed
    onnx package differs from the submodule version, because the generated
    models and reference outputs would not match the recorded expectations.
    """
    submodule_version = _submodule_onnx_version()
    if submodule_version is None:
        return False
    import onnx

    if onnx.__version__ != submodule_version:
        raise NodeTestDataError(
            f"Installed onnx {onnx.__version__} does not match the onnx-org "
            f"submodule ({submodule_version}); install onnx=={submodule_version} "
            "(see requirements-ci.txt) to generate the ONNX node backend tests."
        )
    if _stamped_version(root) == submodule_version:
        return True
    # Build next to the target and swap it in, so that an interrupted run never
    # leaves a stamped but incomplete directory behind.
    staging = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _generate(staging, submodule_version)
        shutil.rmtree(root, ignore_errors=True)
        staging.rename(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        if not ensure_node_test_data():
            print("onnx-org submodule is not initialized.", file=sys.stderr)
            return 1
    except NodeTestDataError as error:
        print(error, file=sys.stderr)
        return 1
    print(NODE_DATA_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
