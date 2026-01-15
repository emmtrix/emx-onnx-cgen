from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnx import TensorProto, helper

from shared.scalar_types import ScalarType

from onnx2c.codegen.c_emitter import MultiInputBinaryOp
from onnx2c.compiler import Compiler, CompilerOptions
from onnx2c.errors import UnsupportedOpError
from onnx2c.lowering.flatten import lower_flatten
from onnx2c.lowering.grid_sample import lower_grid_sample
from onnx2c.lowering.squeeze import lower_squeeze
from onnx2c.lowering import variadic as _variadic  # noqa: F401
from onnx2c.lowering.registry import get_lowering
from onnx2c.onnx_import import import_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_operator_model(
    *,
    op_type: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    dtype: int,
    attrs: dict[str, object] | None = None,
    opset: int = 13,
) -> onnx.ModelProto:
    input_names = [f"in{idx}" for idx in range(len(input_shapes))]
    inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output = helper.make_tensor_value_info("out", dtype, output_shape)
    node = helper.make_node(
        op_type,
        inputs=input_names,
        outputs=[output.name],
        **(attrs or {}),
    )
    graph = helper.make_graph([node], f"{op_type.lower()}_graph", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _flatten_output_shape(input_shape: list[int], axis: int) -> list[int]:
    rank = len(input_shape)
    if axis < 0:
        axis += rank
    first = int(np.prod(input_shape[:axis])) if axis else 1
    second = int(np.prod(input_shape[axis:])) if axis < rank else 1
    return [first, second]


def _make_flatten_model(input_shape: list[int], axis: int) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output_shape = _flatten_output_shape(input_shape, axis)
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "Flatten",
        inputs=["in0"],
        outputs=[output.name],
        axis=axis,
    )
    graph = helper.make_graph(
        [node],
        "flatten_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_squeeze_lowering_model(
    input_shape: list[int],
    output_shape: list[int],
    *,
    axes: list[int] | None = None,
    include_axes_input: bool = False,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    inputs = ["in0"]
    initializers: list[onnx.TensorProto] = []
    attrs: dict[str, object] = {}
    if include_axes_input:
        if axes is None:
            raise ValueError("axes must be provided when axes input is included")
        axes_values = np.array(axes, dtype=np.int64)
        axes_tensor = helper.make_tensor(
            "axes",
            TensorProto.INT64,
            dims=axes_values.shape,
            vals=axes_values.tolist(),
        )
        initializers.append(axes_tensor)
        inputs.append("axes")
    elif axes is not None:
        attrs["axes"] = axes
    node = helper.make_node(
        "Squeeze",
        inputs=inputs,
        outputs=[output.name],
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "squeeze_graph",
        [input_info],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _broadcast_shape(
    input_shape: list[int], target_shape: list[int]
) -> list[int]:
    output_rank = max(len(input_shape), len(target_shape))
    padded_input = [1] * (output_rank - len(input_shape)) + input_shape
    padded_target = [1] * (output_rank - len(target_shape)) + target_shape
    output_shape: list[int] = []
    for input_dim, target_dim in zip(padded_input, padded_target):
        if input_dim == 1:
            output_shape.append(target_dim)
        elif target_dim == 1 or input_dim == target_dim:
            output_shape.append(input_dim)
        else:
            raise ValueError(
                f"Shapes {input_shape} and {target_shape} are not broadcastable"
            )
    return output_shape


def _make_expand_model(
    *,
    input_shape: list[int],
    target_shape: list[int],
    dtype: int,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    output_shape = _broadcast_shape(input_shape, target_shape)
    output = helper.make_tensor_value_info("output", dtype, output_shape)
    shape_tensor = helper.make_tensor(
        "shape",
        TensorProto.INT64,
        dims=[len(target_shape)],
        vals=target_shape,
    )
    node = helper.make_node(
        "Expand", inputs=["input", "shape"], outputs=[output.name]
    )
    graph = helper.make_graph(
        [node],
        "expand_graph",
        [input_info],
        [output],
        initializer=[shape_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_range_model(
    *,
    start: float | int,
    limit: float | int,
    delta: float | int,
    dtype: int,
    opset: int = 11,
) -> onnx.ModelProto:
    start_tensor = helper.make_tensor(
        "start",
        dtype,
        dims=[],
        vals=[start],
    )
    limit_tensor = helper.make_tensor(
        "limit",
        dtype,
        dims=[],
        vals=[limit],
    )
    delta_tensor = helper.make_tensor(
        "delta",
        dtype,
        dims=[],
        vals=[delta],
    )
    output_values = np.arange(start, limit, delta, dtype=np.dtype("float64"))
    output_shape = [int(output_values.shape[0])]
    output = helper.make_tensor_value_info("output", dtype, output_shape)
    node = helper.make_node(
        "Range",
        inputs=["start", "limit", "delta"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "range_graph",
        [],
        [output],
        initializer=[start_tensor, limit_tensor, delta_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_eye_like_model(
    *,
    input_shape: list[int],
    dtype: int,
    k: int = 0,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    output = helper.make_tensor_value_info("output", dtype, input_shape)
    node = helper.make_node(
        "EyeLike",
        inputs=["input"],
        outputs=[output.name],
        k=k,
    )
    graph = helper.make_graph(
        [node], "eye_like_graph", [input_info], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_tile_model(
    *,
    input_shape: list[int],
    repeats: list[int],
    dtype: int,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    output_shape = [dim * repeat for dim, repeat in zip(input_shape, repeats)]
    output = helper.make_tensor_value_info("output", dtype, output_shape)
    repeats_tensor = helper.make_tensor(
        "repeats",
        TensorProto.INT64,
        dims=[len(repeats)],
        vals=repeats,
    )
    node = helper.make_node(
        "Tile",
        inputs=["input", "repeats"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "tile_graph",
        [input_info],
        [output],
        initializer=[repeats_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_pad_model(
    *,
    input_shape: list[int],
    pads: list[int],
    value: float | int | None,
    dtype: int,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    output_shape = [
        dim + pads[index] + pads[index + len(input_shape)]
        for index, dim in enumerate(input_shape)
    ]
    output = helper.make_tensor_value_info("output", dtype, output_shape)
    pads_tensor = helper.make_tensor(
        "pads",
        TensorProto.INT64,
        dims=[len(pads)],
        vals=pads,
    )
    inputs = ["input", "pads"]
    initializers = [pads_tensor]
    if value is not None:
        value_tensor = helper.make_tensor(
            "value",
            dtype,
            dims=[],
            vals=[value],
        )
        inputs.append("value")
        initializers.append(value_tensor)
    node = helper.make_node(
        "Pad",
        inputs=inputs,
        outputs=[output.name],
        mode="constant",
    )
    graph = helper.make_graph(
        [node],
        "pad_graph",
        [input_info],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _depth_to_space_reference(
    value: np.ndarray, *, blocksize: int, mode: str = "DCR"
) -> np.ndarray:
    if value.ndim != 4:
        raise ValueError("DepthToSpace expects 4D input")
    b, c, h, w = value.shape
    if mode == "DCR":
        tmpshape = (
            b,
            blocksize,
            blocksize,
            c // (blocksize * blocksize),
            h,
            w,
        )
        reshaped = value.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
    else:
        tmpshape = (
            b,
            c // (blocksize * blocksize),
            blocksize,
            blocksize,
            h,
            w,
        )
        reshaped = value.reshape(tmpshape)
        transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
    finalshape = (
        b,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    return np.reshape(transposed, finalshape)


def _space_to_depth_reference(
    value: np.ndarray, *, blocksize: int
) -> np.ndarray:
    if value.ndim != 4:
        raise ValueError("SpaceToDepth expects 4D input")
    b, c, h, w = value.shape
    tmpshape = (
        b,
        c,
        h // blocksize,
        blocksize,
        w // blocksize,
        blocksize,
    )
    reshaped = np.reshape(value, tmpshape)
    transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
    finalshape = (
        b,
        c * blocksize * blocksize,
        h // blocksize,
        w // blocksize,
    )
    return np.reshape(transposed, finalshape)


def _make_split_model(
    *,
    input_shape: list[int],
    split_sizes: list[int] | None,
    axis: int,
    dtype: int,
    opset: int = 18,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    outputs: list[onnx.ValueInfoProto] = []
    output_names: list[str] = []
    if split_sizes is None:
        num_outputs = 3
        base = input_shape[axis] // num_outputs
        last = input_shape[axis] - base * (num_outputs - 1)
        split_sizes = [base] * (num_outputs - 1) + [last]
        attrs = {"axis": axis, "num_outputs": num_outputs}
        inputs = ["input"]
        initializer = []
    else:
        attrs = {"axis": axis}
        inputs = ["input", "split"]
        split_tensor = helper.make_tensor(
            "split",
            TensorProto.INT64,
            dims=[len(split_sizes)],
            vals=split_sizes,
        )
        initializer = [split_tensor]
    for index, size in enumerate(split_sizes):
        output_shape = list(input_shape)
        output_shape[axis] = size
        name = f"output_{index}"
        output_names.append(name)
        outputs.append(helper.make_tensor_value_info(name, dtype, output_shape))
    node = helper.make_node(
        "Split",
        inputs=inputs,
        outputs=output_names,
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "split_graph",
        [input_info],
        outputs,
        initializer=initializer,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_compare_model(
    *,
    op_type: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    input_dtype: int,
    opset: int = 13,
) -> onnx.ModelProto:
    input_names = [f"in{idx}" for idx in range(len(input_shapes))]
    inputs = [
        helper.make_tensor_value_info(name, input_dtype, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output = helper.make_tensor_value_info(
        "out", TensorProto.BOOL, output_shape
    )
    node = helper.make_node(
        op_type,
        inputs=input_names,
        outputs=[output.name],
    )
    graph = helper.make_graph([node], f"{op_type.lower()}_graph", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_reduce_model(
    *,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    axes: list[int],
    keepdims: int,
    dtype: int,
    opset: int = 18,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("in0", dtype, input_shape)
    output = helper.make_tensor_value_info("out", dtype, output_shape)
    axes_values = np.array(axes, dtype=np.int64)
    axes_tensor = helper.make_tensor(
        "axes",
        TensorProto.INT64,
        dims=axes_values.shape,
        vals=axes_values.tolist(),
    )
    node = helper.make_node(
        op_type,
        inputs=["in0", "axes"],
        outputs=[output.name],
        keepdims=keepdims,
    )
    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_graph",
        [input_info],
        [output],
        initializer=[axes_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_reduce_model_with_axes_input(
    *,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    axes_shape: list[int],
    keepdims: int,
    dtype: int,
    opset: int = 18,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("in0", dtype, input_shape)
    axes_info = helper.make_tensor_value_info(
        "axes", TensorProto.INT64, axes_shape
    )
    output = helper.make_tensor_value_info("out", dtype, output_shape)
    node = helper.make_node(
        op_type,
        inputs=["in0", "axes"],
        outputs=[output.name],
        keepdims=keepdims,
    )
    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_graph",
        [input_info, axes_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model

def _make_constant_add_model() -> onnx.ModelProto:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shape)
    const_values = np.linspace(0.5, 1.0, num=6, dtype=np.float32).reshape(input_shape)
    const_tensor = helper.make_tensor(
        "const_tensor",
        TensorProto.FLOAT,
        dims=input_shape,
        vals=const_values.flatten().tolist(),
    )
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_out"],
        value=const_tensor,
    )
    const_info = helper.make_tensor_value_info(
        "const_out", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, input_shape)
    add_node = helper.make_node(
        "Add", inputs=["in0", "const_out"], outputs=[output.name]
    )
    graph = helper.make_graph(
        [const_node, add_node],
        "constant_add_graph",
        [input_info],
        [output],
        value_info=[const_info],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_constant_of_shape_model() -> onnx.ModelProto:
    shape_values = np.array([2, 3, 4], dtype=np.int64)
    shape_tensor = helper.make_tensor(
        "shape",
        TensorProto.INT64,
        dims=shape_values.shape,
        vals=shape_values.tolist(),
    )
    value_tensor = helper.make_tensor(
        "fill",
        TensorProto.FLOAT,
        dims=[1],
        vals=[1.25],
    )
    output_shape = shape_values.tolist()
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    node = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=[output.name],
        value=value_tensor,
    )
    graph = helper.make_graph(
        [node],
        "constant_of_shape_graph",
        [],
        [output],
        initializer=[shape_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_reshape_model() -> onnx.ModelProto:
    input_shape = [2, 3, 4]
    output_shape = [2, 12]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    shape_values = np.array([0, -1], dtype=np.int64)
    shape_tensor = helper.make_tensor(
        "shape",
        TensorProto.INT64,
        dims=shape_values.shape,
        vals=shape_values.tolist(),
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "Reshape",
        inputs=["in0", "shape"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "reshape_graph",
        [input_info],
        [output],
        initializer=[shape_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_squeeze_model() -> onnx.ModelProto:
    input_shape = [1, 3, 1, 5]
    output_shape = [3, 5]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    axes_values = np.array([0, 2], dtype=np.int64)
    axes_tensor = helper.make_tensor(
        "axes",
        TensorProto.INT64,
        dims=axes_values.shape,
        vals=axes_values.tolist(),
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "Squeeze",
        inputs=["in0", "axes"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "squeeze_graph",
        [input_info],
        [output],
        initializer=[axes_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_cast_model() -> onnx.ModelProto:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.INT32, input_shape
    )
    node = helper.make_node(
        "Cast",
        inputs=["in0"],
        outputs=[output.name],
        to=TensorProto.INT32,
    )
    graph = helper.make_graph(
        [node],
        "cast_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_castlike_model() -> onnx.ModelProto:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    like_info = helper.make_tensor_value_info(
        "in1", TensorProto.INT32, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.INT32, input_shape
    )
    node = helper.make_node(
        "CastLike",
        inputs=["in0", "in1"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "castlike_graph",
        [input_info, like_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 19)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_lstm_model(
    *,
    seq_length: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    dtype: int,
    include_optional_inputs: bool,
    include_y: bool,
    include_y_h: bool,
    include_y_c: bool,
    layout: int = 0,
) -> onnx.ModelProto:
    x_shape = (
        [seq_length, batch_size, input_size]
        if layout == 0
        else [batch_size, seq_length, input_size]
    )
    inputs = [
        helper.make_tensor_value_info("X", dtype, x_shape),
        helper.make_tensor_value_info(
            "W", dtype, [1, 4 * hidden_size, input_size]
        ),
        helper.make_tensor_value_info(
            "R", dtype, [1, 4 * hidden_size, hidden_size]
        ),
    ]
    input_names = ["X", "W", "R"]
    if include_optional_inputs:
        inputs.extend(
            [
                helper.make_tensor_value_info(
                    "B", dtype, [1, 8 * hidden_size]
                ),
                helper.make_tensor_value_info(
                    "sequence_lens", TensorProto.INT32, [batch_size]
                ),
                helper.make_tensor_value_info(
                    "initial_h", dtype, [1, batch_size, hidden_size]
                ),
                helper.make_tensor_value_info(
                    "initial_c", dtype, [1, batch_size, hidden_size]
                ),
                helper.make_tensor_value_info(
                    "P", dtype, [1, 3 * hidden_size]
                ),
            ]
        )
        input_names.extend(
            ["B", "sequence_lens", "initial_h", "initial_c", "P"]
        )
    outputs = []
    output_names: list[str] = []
    if include_y:
        y_shape = (
            [seq_length, 1, batch_size, hidden_size]
            if layout == 0
            else [batch_size, seq_length, 1, hidden_size]
        )
        outputs.append(
            helper.make_tensor_value_info("Y", dtype, y_shape)
        )
        output_names.append("Y")
    if include_y_h:
        outputs.append(
            helper.make_tensor_value_info(
                "Y_h", dtype, [1, batch_size, hidden_size]
            )
        )
        output_names.append("Y_h")
    if include_y_c:
        outputs.append(
            helper.make_tensor_value_info(
                "Y_c", dtype, [1, batch_size, hidden_size]
            )
        )
        output_names.append("Y_c")
    node = helper.make_node(
        "LSTM",
        inputs=input_names,
        outputs=output_names,
        hidden_size=hidden_size,
        layout=layout,
    )
    graph = helper.make_graph(
        [node],
        "lstm_graph",
        inputs,
        outputs,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _lstm_reference(
    *,
    x: np.ndarray,
    w: np.ndarray,
    r: np.ndarray,
    b: np.ndarray,
    sequence_lens: np.ndarray,
    initial_h: np.ndarray,
    initial_c: np.ndarray,
    p: np.ndarray,
    layout: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if layout == 1:
        x = np.swapaxes(x, 0, 1)
    seq_length, batch_size, _ = x.shape
    hidden_size = r.shape[-1]
    y = np.zeros((seq_length, 1, batch_size, hidden_size), dtype=x.dtype)
    h_prev = initial_h[0].copy()
    c_prev = initial_c[0].copy()
    bias = b[0, : 4 * hidden_size] + b[0, 4 * hidden_size :]
    p_i = p[0, :hidden_size]
    p_o = p[0, hidden_size : 2 * hidden_size]
    p_f = p[0, 2 * hidden_size :]
    for step in range(seq_length):
        active_mask = step < sequence_lens
        if not np.all(active_mask):
            y[step, 0] = np.where(active_mask[:, None], y[step, 0], 0)
        x_t = x[step]
        gates = x_t @ w[0].T + h_prev @ r[0].T + bias
        i, o, f, c = np.split(gates, 4, axis=1)
        i = 1 / (1 + np.exp(-(i + p_i * c_prev)))
        f = 1 / (1 + np.exp(-(f + p_f * c_prev)))
        c_tilde = np.tanh(c)
        c_new = f * c_prev + i * c_tilde
        o = 1 / (1 + np.exp(-(o + p_o * c_new)))
        h_new = o * np.tanh(c_new)
        h_prev = np.where(active_mask[:, None], h_new, h_prev)
        c_prev = np.where(active_mask[:, None], c_new, c_prev)
        y[step, 0] = np.where(active_mask[:, None], h_prev, 0)
    y_h = h_prev.reshape(1, batch_size, hidden_size)
    y_c = c_prev.reshape(1, batch_size, hidden_size)
    if layout == 1:
        y = np.transpose(y, (2, 0, 1, 3))
    return y, y_h, y_c


def _shape_output_shape(
    input_shape: list[int], start: int | None, end: int | None
) -> list[int]:
    rank = len(input_shape)
    start_index = 0 if start is None else start
    end_index = rank if end is None else end
    if start_index < 0:
        start_index += rank
    if end_index < 0:
        end_index += rank
    start_index = max(0, min(start_index, rank))
    end_index = max(0, min(end_index, rank))
    if end_index <= start_index:
        raise ValueError("Shape start must be less than end")
    return [end_index - start_index]


def _make_shape_model(
    *, input_shape: list[int], start: int | None = None, end: int | None = None, opset: int = 13
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output_shape = _shape_output_shape(input_shape, start, end)
    output = helper.make_tensor_value_info(
        "out", TensorProto.INT64, output_shape
    )
    attrs: dict[str, object] = {}
    if start is not None:
        attrs["start"] = start
    if end is not None:
        attrs["end"] = end
    node = helper.make_node(
        "Shape",
        inputs=["in0"],
        outputs=[output.name],
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "shape_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_size_model(*, input_shape: list[int], opset: int = 13) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.INT64, [])
    node = helper.make_node("Size", inputs=["in0"], outputs=[output.name])
    graph = helper.make_graph([node], "size_graph", [input_info], [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_slice_model() -> onnx.ModelProto:
    input_shape = [2, 3, 4]
    output_shape = [2, 3, 1]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    starts_values = np.array([0, 1], dtype=np.int64)
    ends_values = np.array([2, 3], dtype=np.int64)
    axes_values = np.array([0, 2], dtype=np.int64)
    steps_values = np.array([1, 2], dtype=np.int64)
    starts = helper.make_tensor(
        "starts",
        TensorProto.INT64,
        dims=starts_values.shape,
        vals=starts_values.tolist(),
    )
    ends = helper.make_tensor(
        "ends",
        TensorProto.INT64,
        dims=ends_values.shape,
        vals=ends_values.tolist(),
    )
    axes = helper.make_tensor(
        "axes",
        TensorProto.INT64,
        dims=axes_values.shape,
        vals=axes_values.tolist(),
    )
    steps = helper.make_tensor(
        "steps",
        TensorProto.INT64,
        dims=steps_values.shape,
        vals=steps_values.tolist(),
    )
    node = helper.make_node(
        "Slice",
        inputs=["in0", "starts", "ends", "axes", "steps"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "slice_graph",
        [input_info],
        [output],
        initializer=[starts, ends, axes, steps],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_resize_model() -> onnx.ModelProto:
    input_shape = [1, 1, 2, 2]
    output_shape = [1, 1, 4, 4]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    sizes_values = np.array(output_shape, dtype=np.int64)
    sizes_tensor = helper.make_tensor(
        "sizes",
        TensorProto.INT64,
        dims=sizes_values.shape,
        vals=sizes_values.tolist(),
    )
    node = helper.make_node(
        "Resize",
        inputs=["in0", "", "", "sizes"],
        outputs=[output.name],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [node],
        "resize_graph",
        [input_info],
        [output],
        initializer=[sizes_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_gridsample_model(
    *,
    input_shape: list[int],
    grid_shape: list[int],
    output_shape: list[int],
    mode: str = "linear",
    padding_mode: str = "zeros",
    align_corners: int = 0,
    opset: int = 22,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, input_shape
    )
    grid_info = helper.make_tensor_value_info(
        "grid", TensorProto.FLOAT, grid_shape
    )
    output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "GridSample",
        inputs=["x", "grid"],
        outputs=[output.name],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    graph = helper.make_graph(
        [node],
        "gridsample_graph",
        [input_info, grid_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_dropout_model() -> onnx.ModelProto:
    input_shape = [2, 3, 4]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, input_shape
    )
    node = helper.make_node(
        "Dropout",
        inputs=["in0"],
        outputs=[output.name],
    )
    graph = helper.make_graph(
        [node],
        "dropout_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _unsqueeze_output_shape(
    input_shape: list[int], axes: list[int]
) -> list[int]:
    output_rank = len(input_shape) + len(axes)
    normalized_axes = []
    for axis in axes:
        if axis < 0:
            axis += output_rank
        if axis < 0 or axis >= output_rank:
            raise ValueError(f"Axis {axis} out of range for rank {output_rank}")
        normalized_axes.append(axis)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("Axes must be unique")
    normalized_axes = sorted(normalized_axes)
    output_dims = []
    input_index = 0
    for axis in range(output_rank):
        if axis in normalized_axes:
            output_dims.append(1)
        else:
            output_dims.append(input_shape[input_index])
            input_index += 1
    return output_dims


def _reduce_output_shape(
    input_shape: list[int], axes: list[int], keepdims: int
) -> list[int]:
    rank = len(input_shape)
    normalized = []
    for axis in axes:
        if axis < 0:
            axis += rank
        normalized.append(axis)
    if keepdims:
        return [
            1 if axis in normalized else dim
            for axis, dim in enumerate(input_shape)
        ]
    return [dim for axis, dim in enumerate(input_shape) if axis not in normalized]


def _arg_reduce_output_shape(
    input_shape: list[int], axis: int, keepdims: int
) -> list[int]:
    rank = len(input_shape)
    if axis < 0:
        axis += rank
    if keepdims:
        return [
            1 if dim_axis == axis else dim
            for dim_axis, dim in enumerate(input_shape)
        ]
    return [dim for dim_axis, dim in enumerate(input_shape) if dim_axis != axis]


def _make_arg_reduce_model(
    *,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    axis: int,
    keepdims: int,
    select_last_index: int,
    dtype: int,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", dtype, input_shape)
    output = helper.make_tensor_value_info("output", TensorProto.INT64, output_shape)
    node = helper.make_node(
        op_type,
        inputs=["input"],
        outputs=[output.name],
        axis=axis,
        keepdims=keepdims,
        select_last_index=select_last_index,
    )
    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_unsqueeze_model(
    *, input_shape: list[int], axes: list[int], opset: int = 13
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output_shape = _unsqueeze_output_shape(input_shape, axes)
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    if opset >= 13:
        axes_values = np.array(axes, dtype=np.int64)
        axes_tensor = helper.make_tensor(
            "axes",
            TensorProto.INT64,
            dims=axes_values.shape,
            vals=axes_values.tolist(),
        )
        node = helper.make_node(
            "Unsqueeze",
            inputs=["in0", "axes"],
            outputs=[output.name],
        )
        graph = helper.make_graph(
            [node],
            "unsqueeze_graph",
            [input_info],
            [output],
            initializer=[axes_tensor],
        )
    else:
        node = helper.make_node(
            "Unsqueeze",
            inputs=["in0"],
            outputs=[output.name],
            axes=axes,
        )
        graph = helper.make_graph(
            [node],
            "unsqueeze_graph",
            [input_info],
            [output],
        )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_conv_model() -> onnx.ModelProto:
    input_shape = [1, 1, 4, 4]
    weight_shape = [1, 1, 3, 3]
    output_shape = [1, 1, 4, 4]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    weight_values = np.arange(9, dtype=np.float32).reshape(weight_shape)
    weight_tensor = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        dims=weight_shape,
        vals=weight_values.flatten().tolist(),
    )
    bias_values = np.array([0.25], dtype=np.float32)
    bias_tensor = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        dims=[1],
        vals=bias_values.tolist(),
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["in0", "weight", "bias"],
        outputs=[output.name],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph(
        [conv_node],
        "conv_graph",
        [input_info],
        [output],
        initializer=[weight_tensor, bias_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_batchnorm_model() -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    input_shape = [2, 3, 2, 2]
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    params = {
        "scale": np.array([1.0, 1.5, -0.5], dtype=np.float32),
        "bias": np.array([0.0, 0.1, -0.2], dtype=np.float32),
        "mean": np.array([0.5, -0.5, 1.0], dtype=np.float32),
        "var": np.array([0.25, 0.5, 1.5], dtype=np.float32),
    }
    initializers = []
    for name, values in params.items():
        initializers.append(
            helper.make_tensor(
                name,
                TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().tolist(),
            )
        )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, input_shape
    )
    node = helper.make_node(
        "BatchNormalization",
        inputs=["in0", "scale", "bias", "mean", "var"],
        outputs=[output.name],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node],
        "batchnorm_graph",
        [input_info],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model, params


def _make_lp_normalization_model(
    *, input_shape: list[int], axis: int, p: int
) -> onnx.ModelProto:
    return _make_operator_model(
        op_type="LpNormalization",
        input_shapes=[input_shape],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs={"axis": axis, "p": p},
        opset=22,
    )


def _make_instance_normalization_model(
    *, input_shape: list[int], epsilon: float = 1e-5
) -> onnx.ModelProto:
    channels = input_shape[1]
    return _make_operator_model(
        op_type="InstanceNormalization",
        input_shapes=[input_shape, [channels], [channels]],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs={"epsilon": epsilon},
        opset=22,
    )


def _make_group_normalization_model(
    *, input_shape: list[int], num_groups: int, epsilon: float = 1e-5
) -> onnx.ModelProto:
    channels = input_shape[1]
    return _make_operator_model(
        op_type="GroupNormalization",
        input_shapes=[input_shape, [channels], [channels]],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs={"epsilon": epsilon, "num_groups": num_groups},
        opset=21,
    )


def _make_layer_normalization_model(
    *, input_shape: list[int], axis: int, epsilon: float = 1e-5
) -> onnx.ModelProto:
    axis_index = axis if axis >= 0 else axis + len(input_shape)
    normalized_shape = input_shape[axis_index:]
    return _make_operator_model(
        op_type="LayerNormalization",
        input_shapes=[input_shape, normalized_shape, normalized_shape],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs={"axis": axis, "epsilon": epsilon},
        opset=17,
    )


def _make_mean_variance_normalization_model(
    *, input_shape: list[int], axes: list[int] | None = None
) -> onnx.ModelProto:
    attrs = {"axes": axes} if axes is not None else None
    return _make_operator_model(
        op_type="MeanVarianceNormalization",
        input_shapes=[input_shape],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs=attrs,
        opset=13,
    )


def _make_rms_normalization_model(
    *, input_shape: list[int], axis: int, epsilon: float = 1e-5
) -> onnx.ModelProto:
    axis_index = axis if axis >= 0 else axis + len(input_shape)
    normalized_shape = input_shape[axis_index:]
    return _make_operator_model(
        op_type="RMSNormalization",
        input_shapes=[input_shape, normalized_shape],
        output_shape=input_shape,
        dtype=TensorProto.FLOAT,
        attrs={"axis": axis, "epsilon": epsilon},
        opset=23,
    )


def _maxpool_output_shape(
    input_shape: list[int],
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
    ceil_mode: int,
) -> list[int]:
    spatial_rank = len(kernel_shape)
    pad_begin = pads[:spatial_rank]
    pad_end = pads[spatial_rank:]
    out_spatial = []
    for dim, kernel, stride, pad_start, pad_finish in zip(
        input_shape[2:], kernel_shape, strides, pad_begin, pad_end
    ):
        numerator = dim + pad_start + pad_finish - kernel
        if ceil_mode:
            out_dim = (numerator + stride - 1) // stride + 1
            if (out_dim - 1) * stride >= dim + pad_start:
                out_dim -= 1
        else:
            out_dim = numerator // stride + 1
        out_spatial.append(out_dim)
    return [input_shape[0], input_shape[1], *out_spatial]


def _make_maxpool_model(
    *,
    input_shape: list[int],
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
    ceil_mode: int,
    dtype: int = TensorProto.FLOAT,
    opset: int = 13,
) -> onnx.ModelProto:
    output_shape = _maxpool_output_shape(
        input_shape, kernel_shape, strides, pads, ceil_mode
    )
    input_info = helper.make_tensor_value_info("in0", dtype, input_shape)
    output = helper.make_tensor_value_info("out", dtype, output_shape)
    node = helper.make_node(
        "MaxPool",
        inputs=["in0"],
        outputs=[output.name],
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        ceil_mode=ceil_mode,
    )
    graph = helper.make_graph(
        [node],
        "maxpool_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_gather_elements_model(
    *,
    data_shape: list[int],
    indices_shape: list[int],
    axis: int,
    data_dtype: int = TensorProto.FLOAT,
    indices_dtype: int = TensorProto.INT64,
    indices_values: np.ndarray | None = None,
    indices_as_initializer: bool = False,
    opset: int = 13,
) -> onnx.ModelProto:
    data_input = helper.make_tensor_value_info("data", data_dtype, data_shape)
    inputs = [data_input]
    initializers = []
    value_infos = []
    if indices_as_initializer:
        if indices_values is None:
            raise ValueError("indices_values is required for initializer inputs")
        indices_tensor = helper.make_tensor(
            "indices",
            indices_dtype,
            dims=indices_shape,
            vals=indices_values.flatten().tolist(),
        )
        initializers.append(indices_tensor)
        value_infos.append(
            helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
        )
    else:
        inputs.append(
            helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
        )
    output = helper.make_tensor_value_info("out", data_dtype, indices_shape)
    node = helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=[output.name],
        axis=axis,
    )
    graph = helper.make_graph(
        [node],
        "gather_elements_graph",
        inputs,
        [output],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_gather_model(
    *,
    data_shape: list[int],
    indices_shape: list[int],
    axis: int,
    data_dtype: int = TensorProto.FLOAT,
    indices_dtype: int = TensorProto.INT64,
    indices_values: np.ndarray | None = None,
    indices_as_initializer: bool = False,
    opset: int = 13,
) -> onnx.ModelProto:
    data_input = helper.make_tensor_value_info("data", data_dtype, data_shape)
    inputs = [data_input]
    initializers = []
    value_infos = []
    if indices_as_initializer:
        if indices_values is None:
            raise ValueError("indices_values is required for initializer inputs")
        indices_tensor = helper.make_tensor(
            "indices",
            indices_dtype,
            dims=indices_shape,
            vals=indices_values.flatten().tolist(),
        )
        initializers.append(indices_tensor)
        value_infos.append(
            helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
        )
    else:
        inputs.append(
            helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
        )
    axis_index = axis if axis >= 0 else axis + len(data_shape)
    output_shape = (
        data_shape[:axis_index] + indices_shape + data_shape[axis_index + 1 :]
    )
    output = helper.make_tensor_value_info("out", data_dtype, output_shape)
    node = helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=[output.name],
        axis=axis,
    )
    graph = helper.make_graph(
        [node],
        "gather_graph",
        inputs,
        [output],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _average_pool_output_shape(
    input_shape: list[int],
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
) -> list[int]:
    batch, channels, in_h, in_w = input_shape
    pad_top, pad_left, pad_bottom, pad_right = pads
    out_h = (in_h + pad_top + pad_bottom - kernel_shape[0]) // strides[0] + 1
    out_w = (in_w + pad_left + pad_right - kernel_shape[1]) // strides[1] + 1
    return [batch, channels, out_h, out_w]


def _tensorproto_to_dtype(elem_type: int) -> np.dtype:
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.INT16: np.int16,
        TensorProto.INT8: np.int8,
        TensorProto.UINT64: np.uint64,
        TensorProto.UINT32: np.uint32,
        TensorProto.UINT16: np.uint16,
        TensorProto.UINT8: np.uint8,
        TensorProto.BOOL: np.bool_,
    }
    try:
        return np.dtype(mapping[elem_type])
    except KeyError as exc:
        raise ValueError(f"Unsupported elem_type {elem_type}") from exc


def _value_info_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    shape = []
    tensor_shape = value_info.type.tensor_type.shape
    for dim in tensor_shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(1)
    return shape


def _make_random_array(
    rng: np.random.Generator, *, shape: list[int], dtype: np.dtype
) -> np.ndarray:
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.int64).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        return rng.integers(0, 5, size=shape, dtype=np.int64).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        return rng.integers(-5, 5, size=shape, dtype=np.int64).astype(dtype)
    raise ValueError(f"Unsupported dtype {dtype}")


def _run_ort_compare(model: onnx.ModelProto) -> None:
    initializer_names = {init.name for init in model.graph.initializer}
    rng = np.random.default_rng(0)
    inputs: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        elem_type = value_info.type.tensor_type.elem_type
        dtype = _tensorproto_to_dtype(elem_type)
        shape = _value_info_shape(value_info)
        inputs[value_info.name] = _make_random_array(
            rng, shape=shape, dtype=dtype
        )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, inputs)
    compiled = Compiler().run(model, inputs)
    for output_info, ort_output in zip(model.graph.output, ort_outputs):
        output_name = output_info.name
        compiled_output = compiled[output_name]
        if np.issubdtype(compiled_output.dtype, np.floating):
            np.testing.assert_allclose(
                compiled_output,
                ort_output,
                rtol=1e-4,
                atol=1e-5,
            )
        else:
            np.testing.assert_array_equal(compiled_output, ort_output)


def _run_ort_compare_or_skip(
    model: onnx.ModelProto, *, skip_substrings: tuple[str, ...]
) -> None:
    try:
        _run_ort_compare(model)
    except Exception as exc:  # noqa: BLE001 - keep test behavior aligned with CLI skip
        if any(substr in str(exc) for substr in skip_substrings):
            pytest.skip(
                "onnxruntime does not implement this operator in the test "
                "environment."
            )
        raise


def _compile_and_run_testbench(
    model: onnx.ModelProto, *, testbench_inputs: dict[str, np.ndarray]
) -> dict[str, object]:
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        options = CompilerOptions(
            template_dir=PROJECT_ROOT / "templates",
            emit_testbench=True,
            testbench_inputs=testbench_inputs,
        )
        compiler = Compiler(options)
        generated = compiler.compile(model)
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [compiler_cmd, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [str(exe_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(result.stdout)


def _run_testbench_compare(model: onnx.ModelProto) -> None:
    initializer_names = {init.name for init in model.graph.initializer}
    rng = np.random.default_rng(0)
    inputs: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        elem_type = value_info.type.tensor_type.elem_type
        dtype = _tensorproto_to_dtype(elem_type)
        shape = _value_info_shape(value_info)
        inputs[value_info.name] = _make_random_array(
            rng, shape=shape, dtype=dtype
        )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, inputs)
    payload = _compile_and_run_testbench(model, testbench_inputs=inputs)
    outputs_payload = payload.get("outputs", {})
    for output_info, ort_output in zip(model.graph.output, ort_outputs):
        output_payload = outputs_payload.get(output_info.name)
        if output_payload is None:
            raise AssertionError(f"Missing output {output_info.name} in testbench data")
        output_data = np.array(output_payload["data"], dtype=ort_output.dtype)
        output_data = output_data.reshape(ort_output.shape)
        if np.issubdtype(ort_output.dtype, np.floating):
            np.testing.assert_allclose(
                output_data,
                ort_output,
                rtol=1e-4,
                atol=1e-5,
            )
        else:
            np.testing.assert_array_equal(output_data, ort_output)


OPERATOR_CASES = [
    {
        "name": "AndBool",
        "op_type": "And",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.BOOL,
        "attrs": {},
    },
    {
        "name": "NotBool",
        "op_type": "Not",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.BOOL,
        "attrs": {},
    },
    {
        "name": "FlattenAxis1",
        "op_type": "Flatten",
        "input_shapes": [[2, 3, 4]],
        "output_shape": [2, 12],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": 1},
    },
    {
        "name": "AddInt64",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT64,
        "attrs": {},
    },
    {
        "name": "AddInt32",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT32,
        "attrs": {},
    },
    {
        "name": "AddInt16",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT16,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddInt8",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT8,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddUint64",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.UINT64,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddUint32",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.UINT32,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddUint16",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.UINT16,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddUint8",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.UINT8,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "Pow",
        "op_type": "Pow",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Mod",
        "op_type": "Mod",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {"fmod": 1},
        "opset": 13,
    },
    {
        "name": "Min",
        "op_type": "Min",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Max",
        "op_type": "Max",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Mean",
        "op_type": "Mean",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Sum",
        "op_type": "Sum",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "PRelu",
        "op_type": "PRelu",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "MatMul",
        "op_type": "MatMul",
        "input_shapes": [[2, 3], [3, 4]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMulVectorVector",
        "op_type": "MatMul",
        "input_shapes": [[3], [3]],
        "output_shape": [],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMulVectorMatrix",
        "op_type": "MatMul",
        "input_shapes": [[3], [3, 4]],
        "output_shape": [4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMulMatrixVector",
        "op_type": "MatMul",
        "input_shapes": [[2, 3], [3]],
        "output_shape": [2],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMulBatch",
        "op_type": "MatMul",
        "input_shapes": [[2, 3, 4], [2, 4, 5]],
        "output_shape": [2, 3, 5],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMulBroadcastBatch",
        "op_type": "MatMul",
        "input_shapes": [[1, 3, 4], [2, 4, 5]],
        "output_shape": [2, 3, 5],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Gemm",
        "op_type": "Gemm",
        "input_shapes": [[2, 3], [3, 4]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "GemmBiasBroadcast",
        "op_type": "Gemm",
        "input_shapes": [[3, 2], [3, 4], [4]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"transA": 1, "alpha": 0.5, "beta": 1.5},
        "opset": 13,
    },
    {
        "name": "GemmScalarBias",
        "op_type": "Gemm",
        "input_shapes": [[2, 3], [3, 4], []],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"beta": 0.5},
        "opset": 13,
    },
    {
        "name": "GemmTransBColumnBias",
        "op_type": "Gemm",
        "input_shapes": [[2, 3], [4, 3], [2, 1]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"transB": 1, "alpha": 2.0, "beta": 0.75},
        "opset": 13,
    },
    {
        "name": "ConcatAxis0",
        "op_type": "Concat",
        "input_shapes": [[2, 3], [1, 3]],
        "output_shape": [3, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": 0},
    },
    {
        "name": "ConcatAxis1MultipleInputs",
        "op_type": "Concat",
        "input_shapes": [[2, 1, 4], [2, 2, 4], [2, 3, 4]],
        "output_shape": [2, 6, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": 1},
    },
    {
        "name": "ConcatAxisNegative",
        "op_type": "Concat",
        "input_shapes": [[1, 2, 3], [1, 2, 1]],
        "output_shape": [1, 2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": -1},
    },
    {
        "name": "Attention",
        "op_type": "Attention",
        "input_shapes": [[1, 2, 3, 4], [1, 2, 5, 4], [1, 2, 5, 4]],
        "output_shape": [1, 2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 23,
    },
    {
        "name": "Attention3DGQAWithMask",
        "op_type": "Attention",
        "input_shapes": [[1, 4, 16], [1, 6, 8], [1, 6, 10], [4, 6]],
        "output_shape": [1, 4, 20],
        "dtype": TensorProto.FLOAT,
        "attrs": {"q_num_heads": 4, "kv_num_heads": 2},
        "opset": 23,
    },
]

VARIADIC_OP_CASES = [
    {
        "name": "SumFloat",
        "op_type": "Sum",
        "dtype": TensorProto.FLOAT,
        "input_shape": [2, 3],
        "input_count": 3,
    },
    {
        "name": "MeanFloat",
        "op_type": "Mean",
        "dtype": TensorProto.FLOAT,
        "input_shape": [2, 3],
        "input_count": 3,
    },
    {
        "name": "MaxFloat",
        "op_type": "Max",
        "dtype": TensorProto.FLOAT,
        "input_shape": [2, 3],
        "input_count": 3,
    },
    {
        "name": "MinFloat",
        "op_type": "Min",
        "dtype": TensorProto.FLOAT,
        "input_shape": [2, 3],
        "input_count": 3,
    },
    {
        "name": "AndBool",
        "op_type": "And",
        "dtype": TensorProto.BOOL,
        "input_shape": [2, 3],
        "input_count": 2,
    },
    {
        "name": "OrBool",
        "op_type": "Or",
        "dtype": TensorProto.BOOL,
        "input_shape": [2, 3],
        "input_count": 2,
    },
    {
        "name": "XorBool",
        "op_type": "Xor",
        "dtype": TensorProto.BOOL,
        "input_shape": [2, 3],
        "input_count": 2,
    },
    {
        "name": "BitwiseAndInt32",
        "op_type": "BitwiseAnd",
        "dtype": TensorProto.INT32,
        "input_shape": [2, 3],
        "input_count": 2,
        "opset": 18,
    },
    {
        "name": "BitwiseOrInt32",
        "op_type": "BitwiseOr",
        "dtype": TensorProto.INT32,
        "input_shape": [2, 3],
        "input_count": 2,
        "opset": 18,
    },
    {
        "name": "BitwiseXorInt32",
        "op_type": "BitwiseXor",
        "dtype": TensorProto.INT32,
        "input_shape": [2, 3],
        "input_count": 2,
        "opset": 18,
    },
]

COMPARE_CASES = [
    {
        "name": "EqualFloat",
        "op_type": "Equal",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "input_dtype": TensorProto.FLOAT,
        "opset": 13,
    },
    {
        "name": "GreaterFloat",
        "op_type": "Greater",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "input_dtype": TensorProto.FLOAT,
        "opset": 13,
    },
    {
        "name": "GreaterOrEqualFloat",
        "op_type": "GreaterOrEqual",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "input_dtype": TensorProto.FLOAT,
        "opset": 13,
    },
    {
        "name": "LessFloat",
        "op_type": "Less",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "input_dtype": TensorProto.FLOAT,
        "opset": 13,
    },
    {
        "name": "LessOrEqualFloat",
        "op_type": "LessOrEqual",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "input_dtype": TensorProto.FLOAT,
        "opset": 13,
    },
]

REDUCE_CASES = [
    {
        "name": "ReduceSumAxis1KeepDims",
        "op_type": "ReduceSum",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceSumAxis1NoKeepDims",
        "op_type": "ReduceSum",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 0,
    },
    {
        "name": "ReduceMeanAxis1",
        "op_type": "ReduceMean",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceMaxAxis1",
        "op_type": "ReduceMax",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceMinAxis1",
        "op_type": "ReduceMin",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceProdAxis1",
        "op_type": "ReduceProd",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceL1Axis1",
        "op_type": "ReduceL1",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceL2Axis1",
        "op_type": "ReduceL2",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceLogSumAxis1",
        "op_type": "ReduceLogSum",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceLogSumExpAxis1",
        "op_type": "ReduceLogSumExp",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
    {
        "name": "ReduceSumSquareAxis1",
        "op_type": "ReduceSumSquare",
        "input_shape": [2, 3, 4],
        "axes": [1],
        "keepdims": 1,
    },
]

ARG_REDUCE_CASES = [
    {
        "name": "ArgMaxAxis0Keepdims",
        "op_type": "ArgMax",
        "input_shape": [3, 4],
        "axis": 0,
        "keepdims": 1,
        "select_last_index": 0,
    },
    {
        "name": "ArgMinAxis1NoKeepdims",
        "op_type": "ArgMin",
        "input_shape": [2, 5],
        "axis": 1,
        "keepdims": 0,
        "select_last_index": 0,
    },
    {
        "name": "ArgMaxAxisNeg1",
        "op_type": "ArgMax",
        "input_shape": [2, 3, 4],
        "axis": -1,
        "keepdims": 0,
        "select_last_index": 0,
    },
]

AVG_POOL_CASES = [
    {
        "name": "Kernel2Stride2",
        "input_shape": [1, 1, 4, 4],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [0, 0, 0, 0],
        "count_include_pad": 0,
    },
    {
        "name": "Kernel3Stride1Pad",
        "input_shape": [1, 2, 5, 5],
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "count_include_pad": 0,
    },
    {
        "name": "Kernel3x2Stride2x1Pad",
        "input_shape": [1, 1, 5, 4],
        "kernel_shape": [3, 2],
        "strides": [2, 1],
        "pads": [0, 1, 0, 1],
        "count_include_pad": 1,
    },
]

MAXPOOL_CASES = [
    {
        "name": "MaxPool2dBasic",
        "input_shape": [1, 1, 4, 4],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [0, 0, 0, 0],
        "ceil_mode": 0,
    },
    {
        "name": "MaxPool2dPadCeil",
        "input_shape": [1, 1, 3, 3],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [1, 1, 1, 1],
        "ceil_mode": 1,
    },
]

REARRANGE_ORT_CASES = [
    {
        "name": "Identity",
        "model": lambda: _make_operator_model(
            op_type="Identity",
            input_shapes=[[2, 3]],
            output_shape=[2, 3],
            dtype=TensorProto.FLOAT,
        ),
    },
    {
        "name": "EyeLike",
        "model": lambda: _make_eye_like_model(
            input_shape=[3, 4],
            dtype=TensorProto.FLOAT,
            k=0,
        ),
    },
    {
        "name": "Tile",
        "model": lambda: _make_tile_model(
            input_shape=[2, 1],
            repeats=[2, 3],
            dtype=TensorProto.FLOAT,
        ),
    },
    {
        "name": "PadConstant",
        "model": lambda: _make_pad_model(
            input_shape=[2, 3],
            pads=[1, 2, 3, 4],
            value=1.5,
            dtype=TensorProto.FLOAT,
        ),
    },
    {
        "name": "DepthToSpace",
        "model": lambda: _make_operator_model(
            op_type="DepthToSpace",
            input_shapes=[[1, 8, 2, 2]],
            output_shape=[1, 2, 4, 4],
            dtype=TensorProto.FLOAT,
            attrs={"blocksize": 2, "mode": "DCR"},
        ),
    },
    {
        "name": "SpaceToDepth",
        "model": lambda: _make_operator_model(
            op_type="SpaceToDepth",
            input_shapes=[[1, 2, 4, 4]],
            output_shape=[1, 8, 2, 2],
            dtype=TensorProto.FLOAT,
            attrs={"blocksize": 2},
        ),
    },
]

REARRANGE_UNIT_CASES = [
    {
        "name": "Identity",
        "model": REARRANGE_ORT_CASES[0]["model"],
        "input_name": "in0",
        "input_shape": (2, 3),
        "expected": lambda value: value,
    },
    {
        "name": "EyeLike",
        "model": REARRANGE_ORT_CASES[1]["model"],
        "input_name": "input",
        "input_shape": (3, 4),
        "expected": lambda value: np.eye(3, 4, dtype=value.dtype),
    },
    {
        "name": "Tile",
        "model": REARRANGE_ORT_CASES[2]["model"],
        "input_name": "input",
        "input_shape": (2, 1),
        "expected": lambda value: np.tile(value, (2, 3)),
    },
    {
        "name": "PadConstant",
        "model": REARRANGE_ORT_CASES[3]["model"],
        "input_name": "input",
        "input_shape": (2, 3),
        "expected": lambda value: np.pad(
            value, ((1, 3), (2, 4)), mode="constant", constant_values=1.5
        ),
    },
    {
        "name": "DepthToSpace",
        "model": REARRANGE_ORT_CASES[4]["model"],
        "input_name": "in0",
        "input_shape": (1, 8, 2, 2),
        "expected": lambda value: _depth_to_space_reference(
            value, blocksize=2, mode="DCR"
        ),
    },
    {
        "name": "SpaceToDepth",
        "model": REARRANGE_ORT_CASES[5]["model"],
        "input_name": "in0",
        "input_shape": (1, 2, 4, 4),
        "expected": lambda value: _space_to_depth_reference(value, blocksize=2),
    },
]


def test_lower_flatten_axis_default() -> None:
    model = _make_flatten_model([2, 3, 4], axis=1)
    graph = import_onnx(model)
    op = lower_flatten(graph, graph.nodes[0])
    assert op.input_shape == (2, 3, 4)
    assert op.output_shape == (2, 12)
    assert op.dtype == ScalarType.F32


def test_lower_flatten_negative_axis() -> None:
    model = _make_flatten_model([2, 3, 4], axis=-1)
    graph = import_onnx(model)
    op = lower_flatten(graph, graph.nodes[0])
    assert op.output_shape == (6, 4)


def test_lower_pad_dynamic_axes_input() -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3]
    )
    pads_info = helper.make_tensor_value_info("pads", TensorProto.INT64, [4])
    axes_info = helper.make_tensor_value_info("axes", TensorProto.INT64, [2])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 3]
    )
    node = helper.make_node(
        "Pad",
        inputs=["input", "pads", "", "axes"],
        outputs=[output_info.name],
        mode="constant",
    )
    graph = helper.make_graph(
        [node],
        "pad_axes_graph",
        [input_info, pads_info, axes_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    graph = import_onnx(model)
    op = get_lowering("Pad")(graph, graph.nodes[0])
    assert op.pads_input == "pads"
    assert op.pads_shape == (4,)
    assert op.axes_input == "axes"
    assert op.axes_shape == (2,)
    assert op.pads_begin is None
    assert op.pads_end is None


def test_lower_squeeze_axes_input() -> None:
    model = _make_squeeze_lowering_model(
        [1, 3, 1, 5],
        [3, 5],
        axes=[0, 2],
        include_axes_input=True,
    )
    graph = import_onnx(model)
    op = lower_squeeze(graph, graph.nodes[0])
    assert op.input_shape == (1, 3, 1, 5)
    assert op.output_shape == (3, 5)
    assert op.dtype == ScalarType.F32


def test_lower_squeeze_default_axes() -> None:
    model = _make_squeeze_lowering_model([1, 3, 1, 5], [3, 5])
    graph = import_onnx(model)
    op = lower_squeeze(graph, graph.nodes[0])
    assert op.output_shape == (3, 5)


def test_lower_gridsample_builds_spec() -> None:
    model = _make_gridsample_model(
        input_shape=[1, 2, 3, 4],
        grid_shape=[1, 5, 6, 2],
        output_shape=[1, 2, 5, 6],
    )
    graph = import_onnx(model)
    op = lower_grid_sample(graph, graph.nodes[0])
    assert op.spatial_rank == 2
    assert op.input_spatial == (3, 4)
    assert op.output_spatial == (5, 6)
    assert op.mode == "linear"


def test_lower_variadic_sum_uses_multi_input_op() -> None:
    model = _make_operator_model(
        op_type="Sum",
        input_shapes=[[2, 2], [2, 2], [2, 2]],
        output_shape=[2, 2],
        dtype=TensorProto.FLOAT,
    )
    graph = import_onnx(model)
    lowering = get_lowering("Sum")
    assert lowering is not None
    op = lowering(graph, graph.nodes[0])
    assert isinstance(op, MultiInputBinaryOp)
    assert op.inputs == ("in0", "in1", "in2")


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case["name"])
def test_operator_c_testbench_matches_onnxruntime(case: dict[str, object]) -> None:
    model = _make_operator_model(
        op_type=case["op_type"],
        input_shapes=case["input_shapes"],
        output_shape=case["output_shape"],
        dtype=case["dtype"],
        attrs=case["attrs"],
        opset=case.get("opset", 13),
    )
    _run_ort_compare(model)


@pytest.mark.parametrize("case", COMPARE_CASES, ids=lambda case: case["name"])
def test_compare_ops_match_onnxruntime(case: dict[str, object]) -> None:
    model = _make_compare_model(
        op_type=case["op_type"],
        input_shapes=case["input_shapes"],
        output_shape=case["output_shape"],
        input_dtype=case["input_dtype"],
        opset=case.get("opset", 13),
    )
    _run_ort_compare(model)


@pytest.mark.parametrize("case", VARIADIC_OP_CASES, ids=lambda case: case["name"])
def test_variadic_ops_match_onnxruntime(case: dict[str, object]) -> None:
    input_shape = case["input_shape"]
    input_count = case["input_count"]
    model = _make_operator_model(
        op_type=case["op_type"],
        input_shapes=[input_shape for _ in range(input_count)],
        output_shape=input_shape,
        dtype=case["dtype"],
        attrs={},
        opset=case.get("opset", 13),
    )
    _run_ort_compare(model)


@pytest.mark.parametrize("case", REDUCE_CASES, ids=lambda case: case["name"])
def test_reduce_op_matches_onnxruntime(case: dict[str, object]) -> None:
    output_shape = _reduce_output_shape(
        case["input_shape"], case["axes"], case["keepdims"]
    )
    model = _make_reduce_model(
        op_type=case["op_type"],
        input_shape=case["input_shape"],
        output_shape=output_shape,
        axes=case["axes"],
        keepdims=case["keepdims"],
        dtype=TensorProto.FLOAT,
    )
    _run_ort_compare(model)


@pytest.mark.parametrize("case", ARG_REDUCE_CASES, ids=lambda case: case["name"])
def test_arg_reduce_matches_onnxruntime(case: dict[str, object]) -> None:
    output_shape = _arg_reduce_output_shape(
        case["input_shape"], case["axis"], case["keepdims"]
    )
    model = _make_arg_reduce_model(
        op_type=case["op_type"],
        input_shape=case["input_shape"],
        output_shape=output_shape,
        axis=case["axis"],
        keepdims=case["keepdims"],
        select_last_index=case["select_last_index"],
        dtype=TensorProto.FLOAT,
    )
    _run_ort_compare(model)


def test_argmax_select_last_index_matches_numpy() -> None:
    input_shape = [2, 4]
    axis = 1
    keepdims = 1
    output_shape = _arg_reduce_output_shape(input_shape, axis, keepdims)
    model = _make_arg_reduce_model(
        op_type="ArgMax",
        input_shape=input_shape,
        output_shape=output_shape,
        axis=axis,
        keepdims=keepdims,
        select_last_index=1,
        dtype=TensorProto.FLOAT,
    )
    compiler = Compiler()
    data = np.array(
        [[1.0, 3.0, 3.0, 2.0], [0.0, -1.0, -1.0, -2.0]],
        dtype=np.float32,
    )
    outputs = compiler.run(model, {"input": data})
    flipped = np.flip(data, axis=axis)
    expected = data.shape[axis] - 1 - np.argmax(flipped, axis=axis)
    expected = np.expand_dims(expected, axis=axis)
    np.testing.assert_array_equal(outputs["output"], expected.astype(np.int64))


def test_reduce_op_axes_input_matches_numpy() -> None:
    input_shape = [2, 3, 4]
    axes = [1]
    keepdims = 0
    output_shape = _reduce_output_shape(input_shape, axes, keepdims)
    model = _make_reduce_model_with_axes_input(
        op_type="ReduceSum",
        input_shape=input_shape,
        output_shape=output_shape,
        axes_shape=[len(axes)],
        keepdims=keepdims,
        dtype=TensorProto.FLOAT,
    )
    compiler = Compiler()
    rng = np.random.default_rng(0)
    data = rng.standard_normal(input_shape).astype(np.float32)
    outputs = compiler.run(
        model, {"in0": data, "axes": np.array(axes, dtype=np.int64)}
    )
    expected = np.sum(data, axis=tuple(axes), keepdims=bool(keepdims))
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_castlike_matches_onnxruntime() -> None:
    model = _make_castlike_model()
    _run_ort_compare(model)


def test_size_matches_onnxruntime() -> None:
    model = _make_size_model(input_shape=[2, 3, 4])
    _run_ort_compare(model)


def test_expand_matches_onnxruntime() -> None:
    model = _make_expand_model(
        input_shape=[3, 1],
        target_shape=[2, 3, 4],
        dtype=TensorProto.FLOAT,
    )
    _run_ort_compare(model)


def test_range_matches_onnxruntime() -> None:
    model = _make_range_model(
        start=1,
        limit=7,
        delta=2,
        dtype=TensorProto.INT32,
    )
    _run_ort_compare(model)


def test_split_matches_onnxruntime() -> None:
    model = _make_split_model(
        input_shape=[2, 6],
        split_sizes=[2, 4],
        axis=1,
        dtype=TensorProto.FLOAT,
    )
    _run_ort_compare(model)


@pytest.mark.parametrize("case", REARRANGE_ORT_CASES, ids=lambda case: case["name"])
def test_rearrange_ops_match_onnxruntime(case: dict[str, object]) -> None:
    model = case["model"]()
    _run_ort_compare(model)


@pytest.mark.parametrize("case", REARRANGE_UNIT_CASES, ids=lambda case: case["name"])
def test_rearrange_ops_match_numpy(case: dict[str, object]) -> None:
    model = case["model"]()
    compiler = Compiler()
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal(case["input_shape"]).astype(np.float32)
    outputs = compiler.run(model, {case["input_name"]: input_data})
    expected = case["expected"](input_data)
    np.testing.assert_allclose(
        outputs[model.graph.output[0].name],
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_expand_run_matches_numpy() -> None:
    model = _make_expand_model(
        input_shape=[3, 1],
        target_shape=[2, 3, 4],
        dtype=TensorProto.FLOAT,
    )
    compiler = Compiler()
    data = np.arange(3, dtype=np.float32).reshape(3, 1)
    outputs = compiler.run(model, {"input": data})
    expected_shape = _broadcast_shape([3, 1], [2, 3, 4])
    expected = np.broadcast_to(data, expected_shape)
    np.testing.assert_allclose(outputs["output"], expected, rtol=1e-5, atol=1e-6)


def test_range_run_matches_numpy() -> None:
    model = _make_range_model(start=1, limit=7, delta=2, dtype=TensorProto.INT32)
    compiler = Compiler()
    outputs = compiler.run(model, {})
    expected = np.arange(1, 7, 2, dtype=np.int32)
    np.testing.assert_array_equal(outputs["output"], expected)


def test_size_run() -> None:
    model = _make_size_model(input_shape=[2, 3, 4])
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    outputs = compiler.run(model, {"in0": data})
    expected = np.array(data.size, dtype=np.int64)
    np.testing.assert_array_equal(outputs["out"], expected)


def test_constant_of_shape_run() -> None:
    model = _make_constant_of_shape_model()
    compiler = Compiler()
    outputs = compiler.run(model, {})
    expected = np.full((2, 3, 4), 1.25, dtype=np.float32)
    np.testing.assert_allclose(outputs["out"], expected)


def test_split_run_matches_numpy() -> None:
    model = _make_split_model(
        input_shape=[2, 6],
        split_sizes=[2, 4],
        axis=1,
        dtype=TensorProto.FLOAT,
    )
    compiler = Compiler()
    data = np.arange(12, dtype=np.float32).reshape(2, 6)
    outputs = compiler.run(model, {"input": data})
    expected = np.split(data, [2], axis=1)
    np.testing.assert_allclose(outputs["output_0"], expected[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs["output_1"], expected[1], rtol=1e-5, atol=1e-6)


def test_gemm_run_matches_numpy() -> None:
    model = _make_operator_model(
        op_type="Gemm",
        input_shapes=[[3, 2], [4, 3], [4]],
        output_shape=[2, 4],
        dtype=TensorProto.FLOAT,
        attrs={"transA": 1, "transB": 1, "alpha": 0.5, "beta": 1.25},
    )
    compiler = Compiler()
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    b = np.arange(12, dtype=np.float32).reshape(4, 3)
    c = np.arange(4, dtype=np.float32)
    outputs = compiler.run(model, {"in0": a, "in1": b, "in2": c})
    expected = 0.5 * (a.T @ b.T) + 1.25 * c
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", AVG_POOL_CASES, ids=lambda case: case["name"])
def test_average_pool_matches_onnxruntime(case: dict[str, object]) -> None:
    output_shape = _average_pool_output_shape(
        case["input_shape"],
        case["kernel_shape"],
        case["strides"],
        case["pads"],
    )
    model = _make_operator_model(
        op_type="AveragePool",
        input_shapes=[case["input_shape"]],
        output_shape=output_shape,
        dtype=TensorProto.FLOAT,
        attrs={
            "kernel_shape": case["kernel_shape"],
            "strides": case["strides"],
            "pads": case["pads"],
            "count_include_pad": case["count_include_pad"],
        },
    )
    _run_ort_compare(model)


def test_global_average_pool_matches_onnxruntime() -> None:
    input_shape = [1, 2, 4, 3]
    model = _make_operator_model(
        op_type="GlobalAveragePool",
        input_shapes=[input_shape],
        output_shape=[input_shape[0], input_shape[1], 1, 1],
        dtype=TensorProto.FLOAT,
    )
    _run_ort_compare(model)


def test_constant_op_matches_onnxruntime() -> None:
    model = _make_constant_add_model()
    _run_ort_compare(model)


def test_constant_of_shape_matches_onnxruntime() -> None:
    model = _make_constant_of_shape_model()
    _run_ort_compare(model)


def test_gather_elements_matches_onnxruntime() -> None:
    indices_values = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64)
    model = _make_gather_elements_model(
        data_shape=[2, 3],
        indices_shape=[2, 3],
        axis=1,
        indices_values=indices_values,
        indices_as_initializer=True,
    )
    _run_ort_compare(model)


def test_gather_matches_onnxruntime() -> None:
    indices_values = np.array([2, 0], dtype=np.int64)
    model = _make_gather_model(
        data_shape=[2, 3, 4],
        indices_shape=[2],
        axis=1,
        indices_values=indices_values,
        indices_as_initializer=True,
    )
    _run_ort_compare(model)


def test_reshape_op_matches_onnxruntime() -> None:
    model = _make_reshape_model()
    _run_ort_compare(model)


def test_squeeze_op_matches_onnxruntime() -> None:
    model = _make_squeeze_model()
    _run_ort_compare(model)


def test_cast_op_matches_onnxruntime() -> None:
    model = _make_cast_model()
    _run_ort_compare(model)


def test_resize_op_matches_onnxruntime() -> None:
    model = _make_resize_model()
    _run_testbench_compare(model)


def test_gridsample_op_matches_onnxruntime() -> None:
    model = _make_gridsample_model(
        input_shape=[1, 1, 3, 4],
        grid_shape=[1, 2, 3, 2],
        output_shape=[1, 1, 2, 3],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )
    _run_testbench_compare(model)


def test_shape_op_matches_onnxruntime() -> None:
    model = _make_shape_model(input_shape=[2, 3, 4])
    _run_ort_compare(model)


def test_shape_slice_op_matches_onnxruntime() -> None:
    model = _make_shape_model(
        input_shape=[2, 3, 4, 5],
        start=1,
        end=3,
        opset=15,
    )
    _run_ort_compare(model)


def test_slice_op_matches_onnxruntime() -> None:
    model = _make_slice_model()
    _run_ort_compare(model)


def test_dropout_op_matches_onnxruntime() -> None:
    model = _make_dropout_model()
    _run_ort_compare(model)


def test_lstm_op_matches_onnxruntime() -> None:
    model = _make_lstm_model(
        seq_length=2,
        batch_size=2,
        input_size=3,
        hidden_size=4,
        dtype=TensorProto.FLOAT,
        include_optional_inputs=False,
        include_y=True,
        include_y_h=True,
        include_y_c=False,
        layout=0,
    )
    _run_ort_compare(model)


def test_unsqueeze_op_matches_onnxruntime() -> None:
    model = _make_unsqueeze_model(input_shape=[2, 3], axes=[-1], opset=13)
    _run_ort_compare(model)


def test_conv_op_matches_onnxruntime() -> None:
    model = _make_conv_model()
    _run_ort_compare(model)


def test_batchnorm_op_matches_onnxruntime() -> None:
    model, _ = _make_batchnorm_model()
    _run_ort_compare(model)


def test_lp_normalization_op_matches_onnxruntime() -> None:
    model = _make_lp_normalization_model(input_shape=[2, 3], axis=-1, p=1)
    _run_ort_compare_or_skip(
        model,
        skip_substrings=(
            "LpNormalization",
            "NOT_IMPLEMENTED",
        ),
    )


def test_instance_normalization_op_matches_onnxruntime() -> None:
    model = _make_instance_normalization_model(input_shape=[1, 3, 2, 2])
    _run_ort_compare(model)


def test_group_normalization_op_matches_onnxruntime() -> None:
    model = _make_group_normalization_model(
        input_shape=[1, 4, 2, 2], num_groups=2
    )
    _run_ort_compare(model)


def test_layer_normalization_op_matches_onnxruntime() -> None:
    model = _make_layer_normalization_model(input_shape=[2, 3, 4], axis=1)
    _run_ort_compare(model)


def test_mean_variance_normalization_op_matches_onnxruntime() -> None:
    model = _make_mean_variance_normalization_model(
        input_shape=[2, 3, 2, 2]
    )
    _run_ort_compare(model)


def test_rms_normalization_op_matches_onnxruntime() -> None:
    model = _make_rms_normalization_model(input_shape=[2, 3, 4], axis=1)
    _run_ort_compare(model)


@pytest.mark.parametrize("case", MAXPOOL_CASES, ids=lambda case: case["name"])
def test_maxpool_op_matches_onnxruntime(case: dict[str, object]) -> None:
    model = _make_maxpool_model(
        input_shape=case["input_shape"],
        kernel_shape=case["kernel_shape"],
        strides=case["strides"],
        pads=case["pads"],
        ceil_mode=case["ceil_mode"],
    )
    _run_ort_compare(model)


def test_gather_elements_run_matches_numpy() -> None:
    model = _make_gather_elements_model(
        data_shape=[2, 3],
        indices_shape=[2, 3],
        axis=1,
    )
    compiler = Compiler()
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    indices = np.array([[2, 0, 1], [-1, 1, 0]], dtype=np.int64)
    outputs = compiler.run(model, {"data": data, "indices": indices})
    expected = np.take_along_axis(data, indices, axis=1)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_gather_run_matches_numpy() -> None:
    model = _make_gather_model(
        data_shape=[2, 3, 4],
        indices_shape=[2, 1],
        axis=2,
    )
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices = np.array([[1], [-1]], dtype=np.int64)
    outputs = compiler.run(model, {"data": data, "indices": indices})
    expected = np.take(data, indices, axis=2)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_dropout_run_matches_numpy() -> None:
    model = _make_dropout_model()
    compiler = Compiler()
    input_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    outputs = compiler.run(model, {"in0": input_data})
    np.testing.assert_allclose(outputs["out"], input_data, rtol=1e-6, atol=1e-6)


def test_lstm_run_matches_numpy() -> None:
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 3
    model = _make_lstm_model(
        seq_length=seq_length,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        dtype=TensorProto.FLOAT,
        include_optional_inputs=True,
        include_y=True,
        include_y_h=True,
        include_y_c=True,
        layout=0,
    )
    compiler = Compiler()
    x = np.linspace(
        0.1, 1.2, num=seq_length * batch_size * input_size, dtype=np.float32
    ).reshape(seq_length, batch_size, input_size)
    w = np.linspace(
        0.2, 0.8, num=4 * hidden_size * input_size, dtype=np.float32
    ).reshape(1, 4 * hidden_size, input_size)
    r = np.linspace(
        0.3, 0.9, num=4 * hidden_size * hidden_size, dtype=np.float32
    ).reshape(1, 4 * hidden_size, hidden_size)
    b = np.full((1, 8 * hidden_size), 0.05, dtype=np.float32)
    sequence_lens = np.array([3, 2], dtype=np.int32)
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    p = np.full((1, 3 * hidden_size), 0.02, dtype=np.float32)
    outputs = compiler.run(
        model,
        {
            "X": x,
            "W": w,
            "R": r,
            "B": b,
            "sequence_lens": sequence_lens,
            "initial_h": initial_h,
            "initial_c": initial_c,
            "P": p,
        },
    )
    expected_y, expected_y_h, expected_y_c = _lstm_reference(
        x=x,
        w=w,
        r=r,
        b=b,
        sequence_lens=sequence_lens,
        initial_h=initial_h,
        initial_c=initial_c,
        p=p,
        layout=0,
    )
    np.testing.assert_allclose(outputs["Y"], expected_y, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(outputs["Y_h"], expected_y_h, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(outputs["Y_c"], expected_y_c, rtol=1e-4, atol=1e-5)


def test_unsqueeze_run_matches_numpy() -> None:
    model = _make_unsqueeze_model(input_shape=[2, 3], axes=[0, 2], opset=11)
    compiler = Compiler()
    input_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    outputs = compiler.run(model, {"in0": input_data})
    expected = np.expand_dims(np.expand_dims(input_data, axis=0), axis=2)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-6, atol=1e-6)


def test_batchnorm_run_matches_numpy() -> None:
    model, params = _make_batchnorm_model()
    compiler = Compiler()
    input_data = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    outputs = compiler.run(model, {"in0": input_data})
    output = outputs["out"]
    epsilon = 1e-5
    reshape_dims = (1, input_data.shape[1], 1, 1)
    scale = params["scale"].reshape(reshape_dims)
    bias = params["bias"].reshape(reshape_dims)
    mean = params["mean"].reshape(reshape_dims)
    var = params["var"].reshape(reshape_dims)
    expected = (input_data - mean) / np.sqrt(var + epsilon) * scale + bias
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)


def test_lp_normalization_run_matches_numpy() -> None:
    model = _make_lp_normalization_model(input_shape=[2, 3], axis=1, p=2)
    compiler = Compiler()
    data = np.array([[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    outputs = compiler.run(model, {"in0": data})
    denom = np.sqrt(np.sum(data * data, axis=1, keepdims=True))
    expected = data / denom
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_instance_normalization_run_matches_numpy() -> None:
    model = _make_instance_normalization_model(input_shape=[1, 2, 2, 3])
    compiler = Compiler()
    data = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    scale = np.array([1.0, 1.5], dtype=np.float32)
    bias = np.array([0.5, -0.25], dtype=np.float32)
    outputs = compiler.run(model, {"in0": data, "in1": scale, "in2": bias})
    mean = np.mean(data, axis=(2, 3), keepdims=True)
    var = np.mean((data - mean) ** 2, axis=(2, 3), keepdims=True)
    scale_reshaped = scale.reshape(1, 2, 1, 1)
    bias_reshaped = bias.reshape(1, 2, 1, 1)
    expected = (data - mean) / np.sqrt(var + 1e-5) * scale_reshaped + bias_reshaped
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_group_normalization_run_matches_numpy() -> None:
    model = _make_group_normalization_model(
        input_shape=[1, 4, 2, 2], num_groups=2
    )
    compiler = Compiler()
    data = np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2)
    scale = np.array([1.0, 1.5, 0.5, -1.0], dtype=np.float32)
    bias = np.array([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
    outputs = compiler.run(model, {"in0": data, "in1": scale, "in2": bias})
    grouped = data.reshape(1, 2, 2, 2, 2)
    mean = np.mean(grouped, axis=(2, 3, 4), keepdims=True)
    var = np.mean((grouped - mean) ** 2, axis=(2, 3, 4), keepdims=True)
    normalized = (grouped - mean) / np.sqrt(var + 1e-5)
    normalized = normalized.reshape(data.shape)
    scale_reshaped = scale.reshape(1, 4, 1, 1)
    bias_reshaped = bias.reshape(1, 4, 1, 1)
    expected = normalized * scale_reshaped + bias_reshaped
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_layer_normalization_run_matches_numpy() -> None:
    model = _make_layer_normalization_model(
        input_shape=[2, 3, 4], axis=-1
    )
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    scale = np.linspace(0.5, 1.5, num=4, dtype=np.float32)
    bias = np.linspace(-0.5, 0.5, num=4, dtype=np.float32)
    outputs = compiler.run(model, {"in0": data, "in1": scale, "in2": bias})
    mean = np.mean(data, axis=2, keepdims=True)
    var = np.mean((data - mean) ** 2, axis=2, keepdims=True)
    expected = (data - mean) / np.sqrt(var + 1e-5)
    expected = expected * scale.reshape(1, 1, 4) + bias.reshape(1, 1, 4)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_mean_variance_normalization_run_matches_numpy() -> None:
    model = _make_mean_variance_normalization_model(
        input_shape=[2, 3, 2, 2], axes=[0, 2, 3]
    )
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    outputs = compiler.run(model, {"in0": data})
    mean = np.mean(data, axis=(0, 2, 3), keepdims=True)
    var = np.mean((data - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    expected = (data - mean) / np.sqrt(var + 1e-9)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)


def test_rms_normalization_run_matches_numpy() -> None:
    model = _make_rms_normalization_model(input_shape=[2, 3, 4], axis=-1)
    compiler = Compiler()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    scale = np.linspace(0.25, 1.0, num=4, dtype=np.float32)
    outputs = compiler.run(model, {"in0": data, "in1": scale})
    mean_square = np.mean(data * data, axis=2, keepdims=True)
    expected = data / np.sqrt(mean_square + 1e-5)
    expected = expected * scale.reshape(1, 1, 4)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-5, atol=1e-6)
