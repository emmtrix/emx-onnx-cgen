from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

from onnx import TensorProto, helper

from onnx2c.compiler import Compiler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


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


def _run_cli_verify(model: onnx.ModelProto) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx.save_model(model, model_path)
        env = os.environ.copy()
        python_path = str(SRC_ROOT)
        if env.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
        subprocess.run(
            [
                sys.executable,
                "-m",
                "onnx2c",
                "verify",
                str(model_path),
                "--template-dir",
                str(PROJECT_ROOT / "templates"),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )


OPERATOR_CASES = [
    {
        "name": "Add",
        "op_type": "Add",
        "input_shapes": [[2, 3, 4], [2, 3, 4]],
        "output_shape": [2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
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
        "name": "Mul",
        "op_type": "Mul",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Sub",
        "op_type": "Sub",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Div",
        "op_type": "Div",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
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
        "name": "Tanh",
        "op_type": "Tanh",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Abs",
        "op_type": "Abs",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Ceil",
        "op_type": "Ceil",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Cos",
        "op_type": "Cos",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Exp",
        "op_type": "Exp",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Floor",
        "op_type": "Floor",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Log",
        "op_type": "Log",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Neg",
        "op_type": "Neg",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Relu",
        "op_type": "Relu",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
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
    {
        "name": "SoftmaxAxis0",
        "op_type": "Softmax",
        "input_shapes": [[2, 3, 4]],
        "output_shape": [2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": 0},
    },
    {
        "name": "SoftmaxAxis1",
        "op_type": "Softmax",
        "input_shapes": [[2, 3, 4]],
        "output_shape": [2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": 1},
    },
    {
        "name": "SoftmaxAxisNeg1",
        "op_type": "Softmax",
        "input_shapes": [[2, 3, 4]],
        "output_shape": [2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {"axis": -1},
    },
    {
        "name": "Sin",
        "op_type": "Sin",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Sqrt",
        "op_type": "Sqrt",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Tan",
        "op_type": "Tan",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
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
    _run_cli_verify(model)


@pytest.mark.parametrize("case", COMPARE_CASES, ids=lambda case: case["name"])
def test_compare_ops_match_onnxruntime(case: dict[str, object]) -> None:
    model = _make_compare_model(
        op_type=case["op_type"],
        input_shapes=case["input_shapes"],
        output_shape=case["output_shape"],
        input_dtype=case["input_dtype"],
        opset=case.get("opset", 13),
    )
    _run_cli_verify(model)


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
    _run_cli_verify(model)


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
    _run_cli_verify(model)


def test_global_average_pool_matches_onnxruntime() -> None:
    input_shape = [1, 2, 4, 3]
    model = _make_operator_model(
        op_type="GlobalAveragePool",
        input_shapes=[input_shape],
        output_shape=[input_shape[0], input_shape[1], 1, 1],
        dtype=TensorProto.FLOAT,
    )
    _run_cli_verify(model)


def test_constant_op_matches_onnxruntime() -> None:
    model = _make_constant_add_model()
    _run_cli_verify(model)


def test_constant_of_shape_matches_onnxruntime() -> None:
    model = _make_constant_of_shape_model()
    _run_cli_verify(model)


def test_reshape_op_matches_onnxruntime() -> None:
    model = _make_reshape_model()
    _run_cli_verify(model)


def test_resize_op_matches_onnxruntime() -> None:
    model = _make_resize_model()
    _run_cli_verify(model)


def test_shape_op_matches_onnxruntime() -> None:
    model = _make_shape_model(input_shape=[2, 3, 4])
    _run_cli_verify(model)


def test_shape_slice_op_matches_onnxruntime() -> None:
    model = _make_shape_model(
        input_shape=[2, 3, 4, 5],
        start=1,
        end=3,
        opset=15,
    )
    _run_cli_verify(model)


def test_dropout_run_matches_numpy() -> None:
    model = _make_dropout_model()
    compiler = Compiler()
    input_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    outputs = compiler.run(model, {"in0": input_data})
    np.testing.assert_allclose(outputs["out"], input_data, rtol=1e-6, atol=1e-6)


def test_dropout_op_matches_onnxruntime() -> None:
    model = _make_dropout_model()
    _run_cli_verify(model)


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
    _run_cli_verify(model)


def test_unsqueeze_run_matches_numpy() -> None:
    model = _make_unsqueeze_model(input_shape=[2, 3], axes=[0, 2], opset=11)
    compiler = Compiler()
    input_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    outputs = compiler.run(model, {"in0": input_data})
    expected = np.expand_dims(np.expand_dims(input_data, axis=0), axis=2)
    np.testing.assert_allclose(outputs["out"], expected, rtol=1e-6, atol=1e-6)


def test_unsqueeze_op_matches_onnxruntime() -> None:
    model = _make_unsqueeze_model(input_shape=[2, 3], axes=[-1], opset=13)
    _run_cli_verify(model)


def test_conv_op_matches_onnxruntime() -> None:
    model = _make_conv_model()
    _run_cli_verify(model)


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


def test_batchnorm_op_matches_onnxruntime() -> None:
    model, _ = _make_batchnorm_model()
    _run_cli_verify(model)


@pytest.mark.parametrize("case", MAXPOOL_CASES, ids=lambda case: case["name"])
def test_maxpool_op_matches_onnxruntime(case: dict[str, object]) -> None:
    model = _make_maxpool_model(
        input_shape=case["input_shape"],
        kernel_shape=case["kernel_shape"],
        strides=case["strides"],
        pads=case["pads"],
        ceil_mode=case["ceil_mode"],
    )
    _run_cli_verify(model)
