from __future__ import annotations

from ..codegen.c_emitter import ResizeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from .registry import register_lowering

_SUPPORTED_RESIZE_MESSAGE = (
    "Resize supports nearest mode with asymmetric coordinates and floor "
    "rounding using constant scales or sizes for 4D NCHW inputs"
)


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


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_scales(graph: Graph, name: str, rank: int) -> tuple[float, ...]:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if initializer.type.dtype not in {"float", "double"}:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if initializer.type.shape[0] != rank:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    return tuple(float(value) for value in initializer.data.reshape(-1))


def _resolve_sizes(graph: Graph, name: str, rank: int) -> tuple[int, ...]:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if initializer.type.dtype not in {"int64", "int32"}:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if initializer.type.shape[0] != rank:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    return tuple(int(value) for value in initializer.data.reshape(-1))


def _validate_resize_config(node: Node) -> None:
    supported_attrs = {
        "coordinate_transformation_mode",
        "mode",
        "nearest_mode",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    mode = _decode_attr(node.attrs.get("mode"), "nearest")
    coordinate_mode = _decode_attr(
        node.attrs.get("coordinate_transformation_mode"), "half_pixel"
    )
    nearest_mode = _decode_attr(node.attrs.get("nearest_mode"), "round_prefer_floor")
    if mode != "nearest":
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if coordinate_mode != "asymmetric":
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if nearest_mode != "floor":
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)


@register_lowering("Resize")
def lower_resize(graph: Graph, node: Node) -> ResizeOp:
    if len(node.inputs) != 4 or len(node.outputs) != 1:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    _validate_resize_config(node)
    if node.inputs[1]:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Resize expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    scales_name = node.inputs[2]
    sizes_name = node.inputs[3]
    if scales_name and sizes_name:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    if not scales_name and not sizes_name:
        raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
    batch, channels, in_h, in_w = input_shape
    if sizes_name:
        sizes = _resolve_sizes(graph, sizes_name, len(input_shape))
        if sizes[0] != batch or sizes[1] != channels:
            raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
        out_h, out_w = sizes[2], sizes[3]
        if output_shape != sizes:
            raise ShapeInferenceError(
                "Resize output shape must be "
                f"{sizes}, got {output_shape}"
            )
        scale_h = out_h / in_h
        scale_w = out_w / in_w
    else:
        scales = _resolve_scales(graph, scales_name, len(input_shape))
        if scales[0] != 1.0 or scales[1] != 1.0:
            raise UnsupportedOpError(_SUPPORTED_RESIZE_MESSAGE)
        scale_h, scale_w = scales[2], scales[3]
        out_h = int(round(in_h * scale_h))
        out_w = int(round(in_w * scale_w))
        expected_output = (batch, channels, out_h, out_w)
        if output_shape != expected_output:
            raise ShapeInferenceError(
                "Resize output shape must be "
                f"{expected_output}, got {output_shape}"
            )
    if out_h <= 0 or out_w <= 0:
        raise ShapeInferenceError("Resize output shape must be positive")
    return ResizeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        scale_h=scale_h,
        scale_w=scale_w,
        dtype=input_dtype,
    )
