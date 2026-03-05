from __future__ import annotations

from dataclasses import dataclass

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import Col2ImOp
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_int64_initializer(graph: Graph, name: str, label: str) -> tuple[int, ...]:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        raise UnsupportedOpError(
            f"Col2Im requires {label} to be a constant initializer, "
            f"but '{name}' is not available at compile time"
        )
    return tuple(int(v) for v in initializer.data.reshape(-1))


@dataclass(frozen=True)
class Col2ImSpec:
    batch: int
    channels: int
    spatial_rank: int
    image_shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    col_dims: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]


def resolve_col2im_spec(graph: Graph, node: Node) -> Col2ImSpec:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    supported_attrs = {"dilations", "pads", "strides"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Col2Im has unsupported attributes")

    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 3:
        raise UnsupportedOpError(
            f"Col2Im expects 3D input (N, C*prod(block_shape), L), got rank {len(input_shape)}"
        )

    image_shape = _read_int64_initializer(graph, node.inputs[1], "image_shape")
    block_shape = _read_int64_initializer(graph, node.inputs[2], "block_shape")

    spatial_rank = len(image_shape)
    if len(block_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"Col2Im image_shape rank ({spatial_rank}) must match "
            f"block_shape rank ({len(block_shape)})"
        )

    batch = input_shape[0]
    kernel_total = 1
    for b in block_shape:
        kernel_total *= b
    if input_shape[1] % kernel_total != 0:
        raise ShapeInferenceError(
            f"Col2Im input dim 1 ({input_shape[1]}) must be divisible by "
            f"prod(block_shape)={kernel_total}"
        )
    channels = input_shape[1] // kernel_total

    strides = tuple(
        int(v) for v in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im strides rank ({len(strides)}) must match spatial rank ({spatial_rank})"
        )

    dilations = tuple(
        int(v) for v in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im dilations rank ({len(dilations)}) must match spatial rank ({spatial_rank})"
        )

    pads = tuple(
        int(v) for v in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im pads rank ({len(pads)}) must be 2 * spatial rank ({2 * spatial_rank})"
        )

    col_dims: list[int] = []
    for i in range(spatial_rank):
        effective_block = dilations[i] * (block_shape[i] - 1) + 1
        pad_total = pads[i] + pads[i + spatial_rank]
        col_dim = (image_shape[i] + pad_total - effective_block) // strides[i] + 1
        if col_dim <= 0:
            raise ShapeInferenceError(
                f"Col2Im column dimension {i} is non-positive ({col_dim})"
            )
        col_dims.append(col_dim)

    expected_L = 1
    for c in col_dims:
        expected_L *= c
    if input_shape[2] != expected_L:
        raise ShapeInferenceError(
            f"Col2Im input dim 2 ({input_shape[2]}) must equal "
            f"prod(col_dims)={expected_L} (col_dims={col_dims})"
        )

    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, *image_shape)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"Col2Im output shape must be {expected_output_shape}, got {output_shape}"
        )

    return Col2ImSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        image_shape=image_shape,
        block_shape=block_shape,
        col_dims=tuple(col_dims),
        strides=strides,
        pads=pads,
        dilations=dilations,
    )


@register_lowering("Col2Im")
def lower_col2im(graph: Graph, node: Node) -> Col2ImOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "Col2Im supports float16, float, and double inputs only"
        )
    spec = resolve_col2im_spec(graph, node)
    return Col2ImOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        image_shape=spec.image_shape,
        block_shape=spec.block_shape,
        col_dims=spec.col_dims,
        strides=spec.strides,
        pads=spec.pads,
        dilations=spec.dilations,
        dtype=op_dtype,
    )
