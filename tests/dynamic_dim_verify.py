"""Verify models with named dynamic dimensions against ONNX Runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from emx_onnx_cgen import cli


def write_test_data(
    model: onnx.ModelProto,
    data_dir: Path,
    dims: Mapping[str, int],
    rng: np.random.Generator,
) -> None:
    """Write ``input_*.pb`` files, sizing each named dim from ``dims``."""

    data_dir.mkdir(parents=True)
    for index, value_info in enumerate(model.graph.input):
        tensor_type = value_info.type.tensor_type
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dims[dim.dim_param]
            for dim in tensor_type.shape.dim
        ]
        if tensor_type.elem_type == TensorProto.INT64:
            array = np.ones(shape, dtype=np.int64)
        else:
            array = rng.standard_normal(shape).astype(np.float32)
        tensor = numpy_helper.from_array(array, value_info.name)
        (data_dir / f"input_{index}.pb").write_bytes(tensor.SerializeToString())


def verify_for_dims(
    model: onnx.ModelProto,
    tmp_path: Path,
    dim_sets: Sequence[Mapping[str, int]],
    extra_args: Sequence[str] = (),
) -> None:
    """Run ``verify`` against ORT once per entry of ``dim_sets``."""

    onnx.save(model, tmp_path / "model.onnx")
    rng = np.random.default_rng(0)
    for index, dims in enumerate(dim_sets):
        data_dir = f"test_data_set_{index}"
        write_test_data(model, tmp_path / data_dir, dims, rng)
        result = cli.run_cli_command(
            [
                "verify",
                *extra_args,
                "--model-base-dir",
                str(tmp_path),
                "model.onnx",
                "--test-data-dir",
                data_dir,
                "--test-data-inputs-only",
            ]
        )
        assert result.exit_code == 0, (dims, result.result)
        assert result.result is not None and result.result.startswith("OK"), dims
