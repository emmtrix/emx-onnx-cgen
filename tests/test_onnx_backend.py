from __future__ import annotations

import gc
import io
from pathlib import Path
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from emx_onnx_cgen.onnx_backend import backend as backend_module


def _make_identity_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "identity_graph", [input_info], [output_info])
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    return model


def _make_sequence_io_model() -> onnx.ModelProto:
    seq_input = helper.make_tensor_sequence_value_info(
        "seq_in", TensorProto.FLOAT, [2, 3]
    )
    tensor_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    seq_output = helper.make_tensor_sequence_value_info(
        "seq_out", TensorProto.FLOAT, [2, 3]
    )
    tensor_output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    identity_node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [identity_node],
        "sequence_io_graph",
        [seq_input, tensor_input],
        [seq_output, tensor_output],
    )
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    return model


def _make_sequence_length_from_dynamic_split_model() -> onnx.ModelProto:
    tensor_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["n"])
    split_input = helper.make_tensor_value_info("Splits", TensorProto.INT64, [3])
    output = helper.make_tensor_value_info("len", TensorProto.INT64, [])
    seq_info = helper.make_tensor_sequence_value_info("seq_1", TensorProto.FLOAT, ["m"])
    graph = helper.make_graph(
        [
            helper.make_node(
                "SplitToSequence", inputs=["X", "Splits"], outputs=["seq_1"]
            ),
            helper.make_node("SequenceLength", inputs=["seq_1"], outputs=["len"]),
        ],
        "sequence_length_from_dynamic_split",
        [tensor_input, split_input],
        [output],
        value_info=[seq_info],
    )
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 12)],
    )
    model.ir_version = 7
    return model


def test_sequence_inputs_reject_lengths_above_fixed_backend_capacity() -> None:
    model = _make_sequence_io_model()
    metadata = backend_module._build_backend_metadata(model)
    seq_value = metadata.inputs[0]
    oversized_sequence = [
        [[float(index)] * 3, [float(index)] * 3]
        for index in range(backend_module._SEQUENCE_MAX_LEN + 1)
    ]

    with pytest.raises(ValueError, match="supports at most 32 items"):
        backend_module._serialize_runtime_input(
            io.BytesIO(),
            value=seq_value,
            input_data=oversized_sequence,
            lowered_sequence_input_shapes=metadata.lowered_sequence_input_shapes,
        )


def test_prepare_keeps_compiled_artifact_alive_only_for_backend_rep_lifetime() -> None:
    model = _make_identity_model()
    created: dict[str, Path] = {}

    def fake_compile_model(_model: onnx.ModelProto) -> backend_module._CompiledArtifact:
        temp_dir = tempfile.TemporaryDirectory(prefix="emx_onnx_backend_test_")
        temp_path = Path(temp_dir.name)
        executable = temp_path / "model"
        executable.write_text("", encoding="utf-8")
        created["temp_path"] = temp_path
        return backend_module._CompiledArtifact(
            executable=executable, temp_dir=temp_dir
        )

    original_compile_model = backend_module._compile_model
    backend_module._compile_model = fake_compile_model
    try:
        rep = backend_module.EmxOnnxCgenBackend.prepare(model)
        temp_path = created["temp_path"]
        assert temp_path.exists()
        del rep
        gc.collect()
        assert not temp_path.exists()
    finally:
        backend_module._compile_model = original_compile_model


def test_backend_runs_dynamic_split_to_sequence_with_empty_tensor() -> None:
    if backend_module._resolve_compiler(None, prefer_ccache=False) is None:
        pytest.skip("No C compiler available for backend integration test.")

    model = _make_sequence_length_from_dynamic_split_model()

    outputs = backend_module.EmxOnnxCgenBackend.run_model(
        model,
        [
            np.array([], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.int64),
        ],
    )

    assert len(outputs) == 1
    assert int(np.asarray(outputs[0]).item()) == 3
