from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import TensorScatterOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import normalize_axis
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering

_ALLOWED_MODES = {"linear", "circular"}


@register_lowering("TensorScatter")
def lower_tensor_scatter(graph: Graph, node: Node) -> TensorScatterOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "TensorScatter must have 2 or 3 inputs and 1 output"
        )
    past_cache_name = node.inputs[0]
    update_name = node.inputs[1]
    write_indices_name = optional_name(node.inputs, 2)
    output_name = node.outputs[0]
    past_cache_shape = value_shape(graph, past_cache_name, node)
    update_shape = value_shape(graph, update_name, node)
    output_shape = value_shape(graph, output_name, node)
    if output_shape != past_cache_shape:
        raise ShapeInferenceError(
            "TensorScatter output shape must match past_cache shape, "
            f"got {output_shape} vs {past_cache_shape}"
        )
    if len(update_shape) != len(past_cache_shape):
        raise ShapeInferenceError(
            "TensorScatter update shape rank must match past_cache rank, "
            f"got {len(update_shape)} vs {len(past_cache_shape)}"
        )
    axis = normalize_axis(int(node.attrs.get("axis", -2)), past_cache_shape, node)
    if axis == 0:
        raise UnsupportedOpError(
            "TensorScatter axis cannot be 0 (batch dimension)"
        )
    for dim_index, (past_dim, update_dim) in enumerate(
        zip(past_cache_shape, update_shape)
    ):
        if dim_index == axis:
            if update_dim > past_dim:
                raise ShapeInferenceError(
                    "TensorScatter update sequence length must be <= "
                    "past_cache sequence length, "
                    f"got {update_dim} vs {past_dim}"
                )
        elif update_dim != past_dim:
            raise ShapeInferenceError(
                "TensorScatter update shape must match past_cache shape "
                f"outside axis {axis}, got {update_shape} vs {past_cache_shape}"
            )
    mode = node.attrs.get("mode", "linear")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode not in _ALLOWED_MODES:
        raise UnsupportedOpError(
            "TensorScatter mode must be one of "
            f"{sorted(_ALLOWED_MODES)}, got {mode}"
        )
    dtype = value_dtype(graph, past_cache_name, node)
    update_dtype = value_dtype(graph, update_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if update_dtype != dtype or output_dtype != dtype:
        raise UnsupportedOpError(
            "TensorScatter expects past_cache, update, and output "
            "to share the same dtype, "
            f"got {dtype.onnx_name}, {update_dtype.onnx_name}, "
            f"{output_dtype.onnx_name}"
        )
    write_indices_shape = None
    write_indices_dtype = None
    if write_indices_name is not None:
        write_indices_shape = value_shape(graph, write_indices_name, node)
        if len(write_indices_shape) != 1:
            raise ShapeInferenceError(
                "TensorScatter write_indices must be a 1D tensor"
            )
        if write_indices_shape[0] != past_cache_shape[0]:
            raise ShapeInferenceError(
                "TensorScatter write_indices length must match batch size, "
                f"got {write_indices_shape[0]} vs {past_cache_shape[0]}"
            )
        write_indices_dtype = value_dtype(
            graph, write_indices_name, node
        )
        if write_indices_dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                "TensorScatter write_indices must be int32 or int64, "
                f"got {write_indices_dtype.onnx_name}"
            )
    return TensorScatterOp(
        past_cache=past_cache_name,
        update=update_name,
        write_indices=write_indices_name,
        output=output_name,
        past_cache_shape=past_cache_shape,
        update_shape=update_shape,
        output_shape=output_shape,
        write_indices_shape=write_indices_shape,
        axis=axis,
        mode=mode,
        dtype=dtype,
        write_indices_dtype=write_indices_dtype,
    )
