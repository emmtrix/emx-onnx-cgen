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
    axes: tuple[int, ...] | None
    axes_input: str | None
    axes_input_shape: tuple[int, ...] | None
    axes_input_dtype: str | None
    keepdims: bool
    output_shape: tuple[int, ...]
    reduce_count: int | None


@dataclass(frozen=True)
class _AxesInputSpec:
    axes: tuple[int, ...] | None
    input_name: str | None
    input_shape: tuple[int, ...] | None
    input_dtype: str | None
    present: bool


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


def _axes_input_info(graph: Graph, node: Node) -> _AxesInputSpec:
    if len(node.inputs) < 2:
        return _AxesInputSpec(None, None, None, None, False)
    if node.inputs[1] == "":
        return _AxesInputSpec(None, None, None, None, False)
    initializer = _find_initializer(graph, node.inputs[1])
    if initializer is None:
        try:
            value = graph.find_value(node.inputs[1])
        except KeyError as exc:
            raise UnsupportedOpError(
                f"{node.op_type} axes input must be constant or inferable from shapes"
            ) from exc
        if value.type.dtype not in {"int64", "int32"}:
            raise UnsupportedOpError(
                f"{node.op_type} axes input must be int64 or int32"
            )
        if any(dim == 0 for dim in value.type.shape):
            return _AxesInputSpec((), None, None, None, True)
        return _AxesInputSpec(
            None,
            node.inputs[1],
            value.type.shape,
            value.type.dtype,
            True,
        )
    if initializer.type.dtype not in {"int64", "int32"}:
        raise UnsupportedOpError(
            f"{node.op_type} axes input must be int64 or int32"
        )
    data = np.array(initializer.data, dtype=np.int64).ravel()
    return _AxesInputSpec(
        tuple(int(value) for value in data),
        None,
        None,
        None,
        True,
    )


def _infer_axes_from_shapes(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    keepdims: bool,
    node: Node,
) -> tuple[int, ...] | None:
    if keepdims:
        if len(input_shape) != len(output_shape):
            return None
        axes: list[int] = []
        for axis, (in_dim, out_dim) in enumerate(
            zip(input_shape, output_shape)
        ):
            if out_dim == in_dim:
                if in_dim == 1:
                    return None
                continue
            if out_dim == 1 and in_dim != 1:
                axes.append(axis)
                continue
            raise ShapeInferenceError(
                f"{node.op_type} output shape does not match input shape"
            )
        return tuple(axes)
    if len(output_shape) > len(input_shape):
        return None

    results: list[tuple[int, ...]] = []

    def backtrack(
        input_index: int, output_index: int, reduced_axes: list[int]
    ) -> None:
        if output_index == len(output_shape):
            results.append(
                tuple(reduced_axes + list(range(input_index, len(input_shape))))
            )
            return
        if input_index == len(input_shape):
            return
        if input_shape[input_index] == output_shape[output_index]:
            backtrack(input_index + 1, output_index + 1, reduced_axes)
        backtrack(
            input_index + 1, output_index, reduced_axes + [input_index]
        )

    backtrack(0, 0, [])
    unique = {axes for axes in results}
    if len(unique) == 1:
        return tuple(sorted(next(iter(unique))))
    if not unique:
        raise ShapeInferenceError(
            f"{node.op_type} output shape does not match input shape"
        )
    return None


def normalize_reduce_axes(
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
) -> tuple[_ReduceSpec | None, bool]:
    axes_attr = node.attrs.get("axes")
    axes_input = _axes_input_info(graph, node)
    if axes_attr is not None and axes_input.present:
        raise UnsupportedOpError(
            f"{node.op_type} cannot set both axes attribute and axes input"
        )
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    if axes_attr is not None:
        axes = tuple(int(value) for value in axes_attr)
        axes_input_name = None
        axes_input_shape = None
        axes_input_dtype = None
    elif axes_input.axes is not None:
        axes = axes_input.axes
        axes_input_name = None
        axes_input_shape = None
        axes_input_dtype = None
    elif axes_input.present:
        output_shape = _value_shape(graph, node.outputs[0], node)
        axes = _infer_axes_from_shapes(input_shape, output_shape, keepdims, node)
        if axes is None:
            axes_input_name = axes_input.input_name
            axes_input_shape = axes_input.input_shape
            axes_input_dtype = axes_input.input_dtype
        else:
            axes_input_name = None
            axes_input_shape = None
            axes_input_dtype = None
    else:
        axes = ()
        axes_input_name = None
        axes_input_shape = None
        axes_input_dtype = None
    noop_with_empty_axes = bool(int(node.attrs.get("noop_with_empty_axes", 0)))
    if axes is not None and not axes:
        if noop_with_empty_axes:
            return None, True
        axes = tuple(range(len(input_shape)))
    if axes is None:
        output_shape = _value_shape(graph, node.outputs[0], node)
        if keepdims and len(output_shape) != len(input_shape):
            raise ShapeInferenceError(
                f"{node.op_type} output shape rank must match input rank"
            )
        if len(output_shape) > len(input_shape):
            raise ShapeInferenceError(
                f"{node.op_type} output shape rank must not exceed input rank"
            )
        return _ReduceSpec(
            axes=None,
            axes_input=axes_input_name,
            axes_input_shape=axes_input_shape,
            axes_input_dtype=axes_input_dtype,
            keepdims=keepdims,
            output_shape=output_shape,
            reduce_count=None,
        ), False
    axes = normalize_reduce_axes(axes, input_shape, node)
    return _ReduceSpec(
        axes=axes,
        axes_input=None,
        axes_input_shape=None,
        axes_input_dtype=None,
        keepdims=keepdims,
        output_shape=(),
        reduce_count=None,
    ), False


def _resolve_reduce_spec(graph: Graph, node: Node) -> _ReduceSpec | None:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 or 2 inputs and 1 output"
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    axes_spec, noop = resolve_reduce_axes(graph, node, input_shape)
    if noop:
        output_shape = _value_shape(graph, node.outputs[0], node)
        if output_shape != input_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be {input_shape}, got {output_shape}"
            )
        return None
    if axes_spec is None:
        raise ShapeInferenceError(f"{node.op_type} axes spec missing")
    if axes_spec.axes is None:
        return _ReduceSpec(
            axes=None,
            axes_input=axes_spec.axes_input,
            axes_input_shape=axes_spec.axes_input_shape,
            axes_input_dtype=axes_spec.axes_input_dtype,
            keepdims=axes_spec.keepdims,
            output_shape=axes_spec.output_shape,
            reduce_count=None,
        )
    axes = axes_spec.axes
    keepdims = axes_spec.keepdims
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
        axes_input=None,
        axes_input_shape=None,
        axes_input_dtype=None,
        keepdims=keepdims,
        output_shape=output_shape,
        reduce_count=reduce_count,
    )


def _reduce_dtype_supported(dtype: str) -> bool:
    return dtype in {
        "float16",
        "float",
        "double",
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
    if node.op_type in REDUCE_OUTPUTS_FLOAT_ONLY and op_dtype not in {
        "float",
        "double",
        "float16",
    }:
        raise UnsupportedOpError(
            f"{node.op_type} supports float16, float, and double inputs only"
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
            input_dtype=op_dtype,
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    if spec.axes_input and (
        spec.axes_input_shape is None or spec.axes_input_dtype is None
    ):
        raise ShapeInferenceError(
            f"{node.op_type} axes input must have a static shape and dtype"
        )
    return ReduceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=spec.output_shape,
        axes=spec.axes or (),
        axes_input=spec.axes_input,
        axes_input_shape=spec.axes_input_shape,
        axes_input_dtype=spec.axes_input_dtype,
        keepdims=spec.keepdims,
        noop_with_empty_axes=bool(int(node.attrs.get("noop_with_empty_axes", 0))),
        reduce_kind=REDUCE_KIND_BY_OP[node.op_type],
        reduce_count=spec.reduce_count,
        dtype=op_dtype,
    )


for _op_type in REDUCE_KIND_BY_OP:
    register_lowering(_op_type)(lower_reduce)
