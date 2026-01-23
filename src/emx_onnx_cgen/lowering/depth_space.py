from __future__ import annotations

from ..ir.ops import DepthToSpaceOp, SpaceToDepthOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


def _blocksize(node: Node) -> int:
    if "blocksize" not in node.attrs:
        raise UnsupportedOpError(f"{node.op_type} requires blocksize attribute")
    blocksize = int(node.attrs["blocksize"])
    if blocksize <= 0:
        raise UnsupportedOpError(
            f"{node.op_type} blocksize must be > 0, got {blocksize}"
        )
    return blocksize


@register_lowering("DepthToSpace")
def lower_depth_to_space(graph: Graph, node: Node) -> DepthToSpaceOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("DepthToSpace must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise UnsupportedOpError("DepthToSpace only supports 4D inputs")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "DepthToSpace expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    blocksize = _blocksize(node)
    mode_attr = node.attrs.get("mode", "DCR")
    if isinstance(mode_attr, bytes):
        mode = mode_attr.decode()
    else:
        mode = str(mode_attr)
    if mode not in {"DCR", "CRD"}:
        raise UnsupportedOpError(
            "DepthToSpace only supports mode DCR or CRD"
        )
    n, c, h, w = input_shape
    if c % (blocksize * blocksize) != 0:
        raise ShapeInferenceError(
            "DepthToSpace input channels must be divisible by blocksize^2"
        )
    expected_shape = (
        n,
        c // (blocksize * blocksize),
        h * blocksize,
        w * blocksize,
    )
    if output_shape != expected_shape:
        raise ShapeInferenceError(
            "DepthToSpace output shape mismatch: "
            f"expected {expected_shape}, got {output_shape}"
        )
    return DepthToSpaceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        blocksize=blocksize,
        mode=mode,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )


@register_lowering("SpaceToDepth")
def lower_space_to_depth(graph: Graph, node: Node) -> SpaceToDepthOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("SpaceToDepth must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise UnsupportedOpError("SpaceToDepth only supports 4D inputs")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "SpaceToDepth expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    blocksize = _blocksize(node)
    n, c, h, w = input_shape
    if h % blocksize != 0 or w % blocksize != 0:
        raise ShapeInferenceError(
            "SpaceToDepth spatial dims must be divisible by blocksize"
        )
    expected_shape = (
        n,
        c * blocksize * blocksize,
        h // blocksize,
        w // blocksize,
    )
    if output_shape != expected_shape:
        raise ShapeInferenceError(
            "SpaceToDepth output shape mismatch: "
            f"expected {expected_shape}, got {output_shape}"
        )
    return SpaceToDepthOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        blocksize=blocksize,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )
