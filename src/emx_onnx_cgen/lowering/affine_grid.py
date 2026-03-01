from __future__ import annotations

from emx_onnx_cgen.ir.ops import AffineGridOp
from emx_onnx_cgen.errors import ShapeInferenceError, UnsupportedOpError
from emx_onnx_cgen.ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("AffineGrid")
def lower_affine_grid(graph: Graph, node: Node) -> AffineGridOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "AffineGrid expects 2 inputs (theta, size) and 1 output"
        )
    theta_name = node.inputs[0]
    size_name = node.inputs[1]
    grid_name = node.outputs[0]

    theta_shape = value_shape(graph, theta_name, node)
    if len(theta_shape) != 3:
        raise ShapeInferenceError(
            f"AffineGrid theta must have rank 3, got {len(theta_shape)}"
        )
    n = theta_shape[0]
    theta_rows = theta_shape[1]
    theta_cols = theta_shape[2]

    if theta_rows == 2 and theta_cols == 3:
        spatial_rank = 2
    elif theta_rows == 3 and theta_cols == 4:
        spatial_rank = 3
    else:
        raise UnsupportedOpError(
            f"AffineGrid theta shape must be (N, 2, 3) for 2D or (N, 3, 4) for 3D, "
            f"got ({n}, {theta_rows}, {theta_cols})"
        )

    theta_dtype = value_dtype(graph, theta_name, node)
    if not theta_dtype.is_float:
        raise UnsupportedOpError(
            f"AffineGrid theta must be float, got {theta_dtype.onnx_name}"
        )

    align_corners = int(node.attrs.get("align_corners", 0))
    if align_corners not in {0, 1}:
        raise UnsupportedOpError("AffineGrid align_corners must be 0 or 1")

    grid_shape = value_shape(graph, grid_name, node)
    if len(grid_shape) != spatial_rank + 2:
        raise ShapeInferenceError(
            f"AffineGrid grid rank must be {spatial_rank + 2}, got {len(grid_shape)}"
        )
    if grid_shape[0] != n:
        raise ShapeInferenceError(
            f"AffineGrid grid batch dim must be {n}, got {grid_shape[0]}"
        )
    if grid_shape[-1] != spatial_rank:
        raise ShapeInferenceError(
            f"AffineGrid grid last dim must be {spatial_rank}, got {grid_shape[-1]}"
        )
    if any(dim < 0 for dim in grid_shape):
        raise ShapeInferenceError("AffineGrid requires static, non-negative shapes")

    return AffineGridOp(
        theta=theta_name,
        size=size_name,
        grid=grid_name,
        align_corners=bool(align_corners),
        spatial_rank=spatial_rank,
    )
