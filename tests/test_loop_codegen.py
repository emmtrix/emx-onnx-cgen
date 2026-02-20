from __future__ import annotations

import re

import onnx
from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions


def _compile(model_name: str, model) -> str:
    compiler = Compiler(CompilerOptions(model_name=model_name))
    return compiler.compile(model)


def _make_loop_sum_model(*, with_scan: bool) -> onnx.ModelProto:
    trip = helper.make_tensor("trip", TensorProto.INT64, dims=[], vals=[3])
    cond = helper.make_tensor("cond", TensorProto.BOOL, dims=[], vals=[True])

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
    scan = helper.make_tensor_value_info("scan", TensorProto.FLOAT, [3]) if with_scan else None

    # Loop body: sum_out = sum_in + float(iter), cond_out = cond_in.
    iter_in = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [])
    iter_cast = helper.make_node("Cast", inputs=["iter"], outputs=["iter_f"], to=TensorProto.FLOAT)
    add = helper.make_node("Add", inputs=["sum_in", "iter_f"], outputs=["sum_next"])
    cond_id = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_next"])

    body_outputs = [
        helper.make_tensor_value_info("cond_next", TensorProto.BOOL, []),
        helper.make_tensor_value_info("sum_next", TensorProto.FLOAT, []),
    ]
    body_nodes = [iter_cast, add, cond_id]
    if with_scan:
        scan_id = helper.make_node("Identity", inputs=["sum_next"], outputs=["scan_out"])
        body_nodes.append(scan_id)
        body_outputs.append(helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, []))

    body = helper.make_graph(
        body_nodes,
        "loop_body",
        [iter_in, cond_in, sum_in],
        body_outputs,
    )

    loop_inputs = ["trip", "cond", "x"]
    loop_outputs = ["y"] + (["scan"] if with_scan else [])
    loop_node = helper.make_node("Loop", inputs=loop_inputs, outputs=loop_outputs, body=body)

    graph_outputs = [y] + ([scan] if scan is not None else [])
    graph = helper.make_graph(
        [loop_node],
        "loop_sum_graph",
        [x],
        graph_outputs,
        initializer=[trip, cond],
    )
    model = helper.make_model(
        graph,
        producer_name="emx-onnx-cgen",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    return model


def test_codegen_loop_emits_c_for_loop() -> None:
    model = _make_loop_sum_model(with_scan=False)
    source = _compile("loop_sum_model", model)
    assert "for (int64_t idx = 0; idx < trip; ++idx)" in source
    assert "loop_cond_out[0]" in source


def test_codegen_loop_scan_emits_indexed_write() -> None:
    model = _make_loop_sum_model(with_scan=True)
    source = _compile("loop_sum_scan_model", model)
    assert "for (int64_t idx = 0; idx < trip; ++idx)" in source
    # Ensure scan output uses idx-based write.
    assert re.search(r"\[idx \* [^\]]+\+ e\]", source) is not None
