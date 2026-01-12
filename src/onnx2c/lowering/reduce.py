from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codegen.c_emitter import ReduceOp, ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from .registry import register_lowering

REDUCE_KIND_BY_OP = {
    "ReduceSum": "sum",
    "ReduceMean": "mean",
    "ReduceMax": "max",
    "ReduceMin": "min",
    "ReduceProd": "prod",
    "ReduceL1": "l1",
    "ReduceL2": "l2",
    "ReduceLogSum": "logsum",
    "ReduceLogSumExp": "logsumexp",
    "ReduceSumSquare": "sumsquare",
}

REDUCE_OUTPUTS_FLOAT_ONLY = {
    "ReduceMean",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
}


@dataclass(frozen=True)
class _ReduceSpec:
    axes: tuple[int, ...]
    keepdims: bool
    output_shape: tuple[int, ...]
    reduce_count: int


def _value_shape(graph: Graph, name: str, node: Node) -> tuple[int, ...]:
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _value_dtype(graph: Graph, name: str, node: Node) -> str:
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        product *= dim
    return product


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _axes_from_initializer(graph: Graph, node: Node) -> tuple[int, ...] | None:
    if len(node.inputs) < 2:
        return None
    initializer = _find_initializer(graph, node.inputs[1])
    if initializer is None:
        raise UnsupportedOpError(
            f"{node.op_type} axes input must be constant"
        )
    if initializer.type.dtype not in {"int64", "int32"}:
        raise UnsupportedOpError(
            f"{node.op_type} axes input must be int64 or int32"
        )
    data = np.array(initializer.data, dtype=np.int64).ravel()
    return tuple(int(value) for value in data)


def _normalize_axes(
    axes: tuple[int, ...], input_shape: tuple[int, ...], node: Node
) -> tuple[int, ...]:
    rank = len(input_shape)
    normalized: list[int] = []
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ShapeInferenceError(
                f"{node.op_type} axis {axis} is out of range for rank {rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ShapeInferenceError(f"{node.op_type} axes must be unique")
    return tuple(sorted(normalized))


def resolve_reduce_axes(
    graph: Graph, node: Node, input_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], bool]:
    axes_attr = node.attrs.get("axes")
    axes_input = _axes_from_initializer(graph, node)
    if axes_attr is not None and axes_input is not None:
        raise UnsupportedOpError(
            f"{node.op_type} cannot set both axes attribute and axes input"
        )
    if axes_attr is not None:
        axes = tuple(int(value) for value in axes_attr)
    elif axes_input is not None:
        axes = axes_input
    else:
        axes = ()
    noop_with_empty_axes = bool(int(node.attrs.get("noop_with_empty_axes", 0)))
    if not axes:
        if noop_with_empty_axes:
            return (), True
        axes = tuple(range(len(input_shape)))
    axes = _normalize_axes(axes, input_shape, node)
    return axes, False


def _resolve_reduce_spec(graph: Graph, node: Node) -> _ReduceSpec | None:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 or 2 inputs and 1 output"
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    axes, noop = resolve_reduce_axes(graph, node, input_shape)
    if noop:
        output_shape = _value_shape(graph, node.outputs[0], node)
        if output_shape != input_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be {input_shape}, got {output_shape}"
            )
        return None
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    if keepdims:
        output_shape = tuple(
            1 if axis in axes else dim
            for axis, dim in enumerate(input_shape)
        )
    else:
        output_shape = tuple(
            dim
            for axis, dim in enumerate(input_shape)
            if axis not in axes
        )
    expected_output_shape = _value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {output_shape}, got {expected_output_shape}"
        )
    reduce_count = _shape_product(tuple(input_shape[axis] for axis in axes))
    return _ReduceSpec(
        axes=axes,
        keepdims=keepdims,
        output_shape=output_shape,
        reduce_count=reduce_count,
    )


def _reduce_dtype_supported(dtype: str) -> bool:
    return dtype in {
        "float",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    }


def lower_reduce(graph: Graph, node: Node) -> ReduceOp | ReshapeOp:
    if node.op_type not in REDUCE_KIND_BY_OP:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype} and {output_dtype}"
        )
    if not _reduce_dtype_supported(op_dtype):
        raise UnsupportedOpError(
            f"{node.op_type} does not support dtype {op_dtype}"
        )
    if node.op_type in REDUCE_OUTPUTS_FLOAT_ONLY and op_dtype != "float":
        raise UnsupportedOpError(
            f"{node.op_type} supports float inputs only"
        )
    spec = _resolve_reduce_spec(graph, node)
    if spec is None:
        input_shape = _value_shape(graph, node.inputs[0], node)
        output_shape = _value_shape(graph, node.outputs[0], node)
        return ReshapeOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=op_dtype,
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    return ReduceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=spec.output_shape,
        axes=spec.axes,
        keepdims=spec.keepdims,
        reduce_kind=REDUCE_KIND_BY_OP[node.op_type],
        reduce_count=spec.reduce_count,
        dtype=op_dtype,
    )


for _op_type in REDUCE_KIND_BY_OP:
    register_lowering(_op_type)(lower_reduce)
