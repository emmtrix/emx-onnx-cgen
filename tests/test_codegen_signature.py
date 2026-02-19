from __future__ import annotations

import re

from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions


def _signature_param_names(source: str) -> list[str]:
    match = re.search(r"void\s+model\(([^)]*)\)\s*\{", source)
    assert match is not None, "model function signature not found"
    signature = match.group(1)
    names: list[str] = []
    for param in signature.split(","):
        param = param.strip()
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[|$)", param)
        assert name_match is not None, f"param name not found for {param!r}"
        names.append(name_match.group(1))
    return names


def test_compile_dedupes_dim_param_names() -> None:
    input_info = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, ["input0", 2]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, ["input0", 2]
    )
    identity_node = helper.make_node("Identity", inputs=["input0"], outputs=["output0"])
    graph = helper.make_graph(
        [identity_node],
        "dup_dim_param_graph",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph)
    compiler = Compiler(CompilerOptions(model_name="model"))
    source = compiler.compile(model)

    param_names = _signature_param_names(source)
    assert len(param_names) == len(set(param_names))
    assert "int input0_dim" in source


def test_compile_sequence_signature_uses_fixed_capacity_and_count() -> None:
    seq_input = helper.make_tensor_sequence_value_info(
        "seq_in", TensorProto.FLOAT, [2, 3]
    )
    seq_output = helper.make_tensor_sequence_value_info(
        "seq_out", TensorProto.FLOAT, [2, 3]
    )
    tensor_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    tensor_output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    identity_node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [identity_node],
        "sequence_signature_graph",
        [seq_input, tensor_input],
        [seq_output, tensor_output],
    )
    model = helper.make_model(graph)
    compiler = Compiler(CompilerOptions(model_name="model"))
    source = compiler.compile(model)

    assert "#ifndef EMX_SEQUENCE_MAX_LEN" in source
    assert "const float seq_in[EMX_SEQUENCE_MAX_LEN][2][3]" in source
    assert "idx_t seq_in__count" in source
    assert "float seq_out[EMX_SEQUENCE_MAX_LEN][2][3]" in source
    assert "idx_t *seq_out__count" in source
