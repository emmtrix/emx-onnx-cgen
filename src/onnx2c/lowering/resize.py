from __future__ import annotations

from dataclasses import dataclass

from ..codegen.c_emitter import ResizeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from .registry import register_lowering

_SUPPORTED_COORD_MODES = {
    "half_pixel",
    "half_pixel_symmetric",
    "asymmetric",
    "align_corners",
    "pytorch_half_pixel",
    "tf_crop_and_resize",
}
_SUPPORTED_MODES = {"nearest", "linear", "cubic"}
_SUPPORTED_NEAREST_MODES = {
    "round_prefer_floor",
    "round_prefer_ceil",
    "floor",
    "ceil",
}
_SUPPORTED_KEEP_ASPECT = {"stretch", "not_larger", "not_smaller"}


@dataclass(frozen=True)
class _ResizeInputs:
    roi: str | None
    scales: str | None
    sizes: str | None


@dataclass(frozen=True)
class _ResolvedScales:
    scales: tuple[float, ...]
    output_shape: tuple[int, ...]
    axes: tuple[int, ...]


@dataclass(frozen=True)
class _InputConfig:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str


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


def _normalize_axes(
    axes: tuple[int, ...], rank: int, node: Node
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ShapeInferenceError(
                f"Resize axis {axis} is out of range for rank {rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ShapeInferenceError("Resize axes must be unique")
    return tuple(normalized)


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _parse_input_names(node: Node) -> _ResizeInputs:
    inputs = list(node.inputs)
    if len(inputs) > 4:
        raise UnsupportedOpError("Resize expects at most 4 inputs")
    while len(inputs) < 4:
        inputs.append("")
    _, roi, scales, sizes = inputs[:4]
    return _ResizeInputs(
        roi=roi or None,
        scales=scales or None,
        sizes=sizes or None,
    )


def _parse_axes(node: Node, rank: int) -> tuple[int, ...]:
    axes_attr = node.attrs.get("axes")
    if axes_attr is None:
        return tuple(range(rank))
    axes = tuple(int(value) for value in axes_attr)
    return _normalize_axes(axes, rank, node)


def _resolve_input_shapes(
    graph: Graph, node: Node, input_name: str
) -> _InputConfig:
    input_shape = _value_shape(graph, input_name, node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    input_dtype = _value_dtype(graph, input_name, node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    return _InputConfig(
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


def _resolve_scales_from_sizes(
    sizes: tuple[int, ...],
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keep_aspect_ratio_policy: str,
) -> _ResolvedScales:
    rank = len(input_shape)
    full_sizes = list(input_shape)
    for index, axis in enumerate(axes):
        full_sizes[axis] = sizes[index]
    if keep_aspect_ratio_policy != "stretch":
        scales = [full_sizes[axis] / input_shape[axis] for axis in axes]
        if keep_aspect_ratio_policy == "not_larger":
            scale = min(scales)
        else:
            scale = max(scales)
        for axis in axes:
            full_sizes[axis] = _round_half_up(scale * input_shape[axis])
        return _ResolvedScales(
            scales=tuple(
                scale if axis in axes else 1.0
                for axis in range(rank)
            ),
            output_shape=tuple(full_sizes),
            axes=axes,
        )
    scales = tuple(
        full_sizes[axis] / input_shape[axis] if axis in axes else 1.0
        for axis in range(rank)
    )
    return _ResolvedScales(
        scales=scales,
        output_shape=tuple(full_sizes),
        axes=axes,
    )


def _resolve_scales_from_values(
    scales: tuple[float, ...],
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> _ResolvedScales:
    rank = len(input_shape)
    full_scales = [1.0] * rank
    for index, axis in enumerate(axes):
        full_scales[axis] = scales[index]
    output_shape = tuple(
        int(input_shape[axis] * full_scales[axis])
        if axis in axes
        else input_shape[axis]
        for axis in range(rank)
    )
    return _ResolvedScales(
        scales=tuple(full_scales),
        output_shape=output_shape,
        axes=axes,
    )


def _load_initializer_values(
    graph: Graph, name: str, node: Node
) -> tuple[float | int, ...] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    data = initializer.data.reshape(-1)
    return tuple(data.tolist())


def _validate_tensor_1d(
    graph: Graph,
    name: str,
    node: Node,
    dtype_options: set[str],
) -> tuple[int, str]:
    shape = _value_shape(graph, name, node)
    if len(shape) != 1:
        raise UnsupportedOpError("Resize expects 1D auxiliary inputs")
    dtype = _value_dtype(graph, name, node)
    if dtype not in dtype_options:
        raise UnsupportedOpError(
            f"Resize expects {name} to have dtype in {sorted(dtype_options)}"
        )
    return shape[0], dtype


def _resolve_scales(
    graph: Graph,
    node: Node,
    config: _InputConfig,
    inputs: _ResizeInputs,
    axes: tuple[int, ...],
    keep_aspect_ratio_policy: str,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    rank = len(config.input_shape)
    if inputs.scales:
        scale_len, _ = _validate_tensor_1d(
            graph, inputs.scales, node, {"float", "double"}
        )
        if scale_len not in {len(axes), rank}:
            raise UnsupportedOpError("Resize scales length mismatch")
        if scale_len == rank and axes != tuple(range(rank)):
            raise UnsupportedOpError(
                "Resize scales length conflicts with axes configuration"
            )
        scale_axes = axes if scale_len == len(axes) else tuple(range(rank))
        values = _load_initializer_values(graph, inputs.scales, node)
        if values is None:
            scales = tuple(
                config.output_shape[axis] / config.input_shape[axis]
                if axis in scale_axes
                else 1.0
                for axis in range(rank)
            )
            return scales, config.output_shape
        resolved = _resolve_scales_from_values(
            tuple(float(value) for value in values),
            config.input_shape,
            scale_axes,
        )
        return resolved.scales, resolved.output_shape
    if inputs.sizes:
        size_len, _ = _validate_tensor_1d(
            graph, inputs.sizes, node, {"int64", "int32"}
        )
        if size_len not in {len(axes), rank}:
            raise UnsupportedOpError("Resize sizes length mismatch")
        if size_len == rank and axes != tuple(range(rank)):
            raise UnsupportedOpError(
                "Resize sizes length conflicts with axes configuration"
            )
        size_axes = axes if size_len == len(axes) else tuple(range(rank))
        values = _load_initializer_values(graph, inputs.sizes, node)
        if values is None:
            scales = tuple(
                config.output_shape[axis] / config.input_shape[axis]
                if axis in size_axes
                else 1.0
                for axis in range(rank)
            )
            return scales, config.output_shape
        resolved = _resolve_scales_from_sizes(
            tuple(int(value) for value in values),
            config.input_shape,
            size_axes,
            keep_aspect_ratio_policy,
        )
        return resolved.scales, resolved.output_shape
    raise UnsupportedOpError("Resize expects scales or sizes input")


def _validate_output_shape(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
) -> None:
    if expected != actual:
        raise ShapeInferenceError(
            f"Resize output shape must be {expected}, got {actual}"
        )
    if any(dim <= 0 for dim in actual):
        raise ShapeInferenceError("Resize output shape must be positive")


@register_lowering("Resize")
def lower_resize(graph: Graph, node: Node) -> ResizeOp:
    if len(node.outputs) != 1:
        raise UnsupportedOpError("Resize expects one output")
    inputs = _parse_input_names(node)
    if inputs.scales and inputs.sizes:
        raise UnsupportedOpError("Resize cannot set both scales and sizes")
    if not inputs.scales and not inputs.sizes:
        raise UnsupportedOpError("Resize expects scales or sizes input")
    mode = _decode_attr(node.attrs.get("mode"), "nearest")
    coordinate_mode = _decode_attr(
        node.attrs.get("coordinate_transformation_mode"), "half_pixel"
    )
    nearest_mode = _decode_attr(
        node.attrs.get("nearest_mode"), "round_prefer_floor"
    )
    keep_aspect_ratio_policy = _decode_attr(
        node.attrs.get("keep_aspect_ratio_policy"), "stretch"
    )
    antialias = bool(int(node.attrs.get("antialias", 0)))
    cubic_coeff_a = float(node.attrs.get("cubic_coeff_a", -0.75))
    exclude_outside = bool(int(node.attrs.get("exclude_outside", 0)))
    extrapolation_value = float(node.attrs.get("extrapolation_value", 0.0))
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(f"Resize mode {mode!r} is not supported")
    if coordinate_mode not in _SUPPORTED_COORD_MODES:
        raise UnsupportedOpError(
            "Resize coordinate_transformation_mode "
            f"{coordinate_mode!r} is not supported"
        )
    if nearest_mode not in _SUPPORTED_NEAREST_MODES:
        raise UnsupportedOpError(
            f"Resize nearest_mode {nearest_mode!r} is not supported"
        )
    if keep_aspect_ratio_policy not in _SUPPORTED_KEEP_ASPECT:
        raise UnsupportedOpError(
            "Resize keep_aspect_ratio_policy "
            f"{keep_aspect_ratio_policy!r} is not supported"
        )
    if antialias and mode == "nearest":
        raise UnsupportedOpError("Resize antialias is not supported for nearest")
    config = _resolve_input_shapes(graph, node, node.inputs[0])
    if config.input_dtype != config.output_dtype:
        raise UnsupportedOpError(
            "Resize expects matching input/output dtypes, "
            f"got {config.input_dtype} and {config.output_dtype}"
        )
    rank = len(config.input_shape)
    axes = _parse_axes(node, rank)
    scales, expected_output = _resolve_scales(
        graph,
        node,
        config,
        inputs,
        axes,
        keep_aspect_ratio_policy,
    )
    _validate_output_shape(expected_output, config.output_shape)
    roi_shape = None
    roi_axes = None
    roi_dtype = None
    if inputs.roi:
        roi_len, roi_dtype = _validate_tensor_1d(
            graph, inputs.roi, node, {"float", "double"}
        )
        if roi_len == 2 * rank:
            roi_shape = (roi_len,)
        elif roi_len == 2 * len(axes):
            roi_shape = (roi_len,)
            roi_axes = axes
        else:
            raise UnsupportedOpError("Resize roi length mismatch")
        if coordinate_mode != "tf_crop_and_resize" and roi_len != 0:
            roi_axes = roi_axes if roi_len == 2 * len(axes) else None
    if coordinate_mode == "tf_crop_and_resize" and not inputs.roi:
        raise UnsupportedOpError("Resize requires roi for tf_crop_and_resize")
    scales_shape = None
    sizes_shape = None
    scales_dtype = None
    sizes_dtype = None
    scales_axes = None
    sizes_axes = None
    if inputs.scales:
        scale_len, scales_dtype = _validate_tensor_1d(
            graph, inputs.scales, node, {"float", "double"}
        )
        scales_shape = (scale_len,)
        if scale_len == len(axes) and len(axes) != rank:
            scales_axes = axes
    if inputs.sizes:
        size_len, sizes_dtype = _validate_tensor_1d(
            graph, inputs.sizes, node, {"int64", "int32"}
        )
        sizes_shape = (size_len,)
        if size_len == len(axes) and len(axes) != rank:
            sizes_axes = axes
    return ResizeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=config.input_shape,
        output_shape=config.output_shape,
        scales=scales,
        scales_input=inputs.scales,
        sizes_input=inputs.sizes,
        roi_input=inputs.roi,
        axes=axes,
        scales_shape=scales_shape,
        sizes_shape=sizes_shape,
        roi_shape=roi_shape,
        scales_dtype=scales_dtype,
        sizes_dtype=sizes_dtype,
        roi_dtype=roi_dtype,
        scales_axes=scales_axes,
        sizes_axes=sizes_axes,
        roi_axes=roi_axes,
        mode=mode,
        coordinate_transformation_mode=coordinate_mode,
        nearest_mode=nearest_mode,
        cubic_coeff_a=cubic_coeff_a,
        exclude_outside=exclude_outside,
        extrapolation_value=extrapolation_value,
        antialias=antialias,
        keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        dtype=config.input_dtype,
    )
