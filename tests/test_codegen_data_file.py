from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions


def test_compile_with_data_file_emits_externs() -> None:
    input_info = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [2, 2]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [2, 2]
    )
    weights_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weights_initializer = helper.make_tensor(
        name="weights",
        data_type=TensorProto.FLOAT,
        dims=weights_array.shape,
        vals=weights_array.flatten().tolist(),
    )
    add_node = helper.make_node(
        "Add", inputs=["input0", "weights"], outputs=["output0"]
    )
    graph = helper.make_graph(
        [add_node],
        "const_data_graph",
        [input_info],
        [output_info],
        [weights_initializer],
    )
    model = helper.make_model(graph)
    compiler = Compiler(
        CompilerOptions(template_dir=Path("templates"), model_name="const_data")
    )
    main_source, data_source = compiler.compile_with_data_file(model)

    assert "extern const float weights[2][2];" in main_source
    assert "static const float" not in main_source
    assert "const float weights[2][2]" in data_source
