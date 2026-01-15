from __future__ import annotations

import re
from pathlib import Path

from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions


def _signature_param_names(source: str) -> list[str]:
    match = re.search(r"void\s+model\(([^)]*)\)\s*\{", source)
    assert match is not None, "model function signature not found"
    signature = match.group(1)
    names: list[str] = []
    for param in signature.split(","):
        param = param.strip()
        name_match = re.search(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[|$)", param
        )
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
    identity_node = helper.make_node(
        "Identity", inputs=["input0"], outputs=["output0"]
    )
    graph = helper.make_graph(
        [identity_node],
        "dup_dim_param_graph",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph)
    compiler = Compiler(
        CompilerOptions(template_dir=Path("templates"), model_name="model")
    )
    source = compiler.compile(model)

    param_names = _signature_param_names(source)
    assert len(param_names) == len(set(param_names))
    assert "int input0_dim" in source
