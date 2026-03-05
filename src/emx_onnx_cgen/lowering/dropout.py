from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import DropoutOp
from .common import optional_name, value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _is_value_used(graph: Graph, name: str) -> bool:
    if any(value.name == name for value in graph.outputs):
        return True
    return any(name in node.inputs for node in graph.nodes)


@register_lowering("Dropout")
def lower_dropout(graph: Graph, node: Node) -> DropoutOp:
    if len(node.outputs) not in {1, 2}:
        raise UnsupportedOpError("Dropout supports 1 or 2 outputs")
    if len(node.inputs) > 3:
        raise UnsupportedOpError("Dropout supports at most 3 inputs")

    input_name = node.inputs[0]
    ratio_name = optional_name(node.inputs, 1)
    training_mode_name = optional_name(node.inputs, 2)
    output_name = node.outputs[0]
    mask_name = optional_name(node.outputs, 1)

    input_shape = _value_shape(graph, input_name, node)
    output_shape = _value_shape(graph, output_name, node)
    if input_shape != output_shape:
        raise ShapeInferenceError(
            "Dropout output shape must match input shape, "
            f"got {output_shape} for input {input_shape}"
        )

    input_dtype = _value_dtype(graph, input_name, node)
    output_dtype = _value_dtype(graph, output_name, node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Dropout expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            f"Dropout input/output dtype must be float, got {input_dtype.onnx_name}"
        )

    if ratio_name is not None:
        ratio_shape = _value_shape(graph, ratio_name, node)
        if ratio_shape not in {(), (1,)}:
            raise UnsupportedOpError(
                "Dropout ratio input must be a scalar or size-1 tensor, "
                f"got shape {ratio_shape}"
            )
        ratio_dtype = _value_dtype(graph, ratio_name, node)
        if ratio_dtype not in {
            ScalarType.BF16,
            ScalarType.F16,
            ScalarType.F32,
            ScalarType.F64,
        }:
            raise UnsupportedOpError(
                "Dropout ratio dtype must be floating-point, "
                f"got {ratio_dtype.onnx_name}"
            )

    if training_mode_name is not None:
        training_shape = _value_shape(graph, training_mode_name, node)
        if training_shape not in {(), (1,)}:
            raise UnsupportedOpError(
                "Dropout training_mode input must be a scalar or size-1 tensor, "
                f"got shape {training_shape}"
            )
        training_dtype = _value_dtype(graph, training_mode_name, node)
        if training_dtype is not ScalarType.BOOL:
            raise UnsupportedOpError(
                "Dropout training_mode dtype must be bool, "
                f"got {training_dtype.onnx_name}"
            )

    if mask_name is not None and _is_value_used(graph, mask_name):
        mask_shape = _value_shape(graph, mask_name, node)
        if mask_shape != input_shape:
            raise ShapeInferenceError(
                "Dropout mask shape must match input shape, "
                f"got {mask_shape} for input {input_shape}"
            )
        mask_dtype = _value_dtype(graph, mask_name, node)
        if mask_dtype is not ScalarType.BOOL:
            raise UnsupportedOpError(
                f"Dropout mask dtype must be bool, got {mask_dtype.onnx_name}"
            )
    else:
        mask_name = None

    seed_value = node.attrs.get("seed")
    seed = int(seed_value) if seed_value is not None else None
    return DropoutOp(
        input0=input_name,
        ratio=ratio_name,
        training_mode=training_mode_name,
        output=output_name,
        mask=mask_name,
        seed=seed,
    )
