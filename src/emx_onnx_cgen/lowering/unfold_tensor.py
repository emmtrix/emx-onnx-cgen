from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import UnfoldTensorOp
from .common import node_dtype, value_shape
from .registry import register_lowering


@register_lowering("UnfoldTensor")
def lower_unfold_tensor(graph: Graph, node: Node) -> UnfoldTensorOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("UnfoldTensor must have exactly 1 input and 1 output")

    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    input_shape = value_shape(graph, node.inputs[0], node)
    ndim = len(input_shape)

    dim = int(node.attrs.get("dim", 0))
    size = int(node.attrs.get("size"))
    step = int(node.attrs.get("step", 1))

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ShapeInferenceError(
            f"UnfoldTensor dim {dim} out of range for input rank {ndim}"
        )
    if size <= 0:
        raise UnsupportedOpError("UnfoldTensor size must be positive")
    if step <= 0:
        raise UnsupportedOpError("UnfoldTensor step must be positive")

    d_dim = input_shape[dim]
    if size > d_dim:
        raise ShapeInferenceError(
            f"UnfoldTensor size {size} > input dim {d_dim} at dim {dim}"
        )

    n_windows = (d_dim - size) // step + 1
    output_shape = (
        *input_shape[:dim],
        n_windows,
        *input_shape[dim + 1 :],
        size,
    )

    expected_output_shape = value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            f"UnfoldTensor output shape mismatch: expected {output_shape}, "
            f"got {expected_output_shape}"
        )

    return UnfoldTensorOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dim=dim,
        size=size,
        step=step,
        dtype=op_dtype,
    )
