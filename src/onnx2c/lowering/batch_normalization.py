from __future__ import annotations

from dataclasses import dataclass

from ..codegen.c_emitter import BatchNormOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@dataclass(frozen=True)
class _BatchNormSpec:
    shape: tuple[int, ...]
    channels: int
    epsilon: float


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


def _node_dtype(graph: Graph, node: Node, *names: str) -> str:
    dtypes = {_value_dtype(graph, name, node) for name in names}
    if len(dtypes) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching dtypes, got {', '.join(sorted(dtypes))}"
        )
    return next(iter(dtypes))


def _resolve_batch_norm_spec(graph: Graph, node: Node) -> _BatchNormSpec:
    if len(node.inputs) != 5 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "BatchNormalization must have 5 inputs and 1 output"
        )
    supported_attrs = {
        "epsilon",
        "is_test",
        "momentum",
        "spatial",
        "training_mode",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("BatchNormalization has unsupported attributes")
    is_test = int(node.attrs.get("is_test", 1))
    if is_test != 1:
        raise UnsupportedOpError("BatchNormalization supports is_test=1 only")
    training_mode = int(node.attrs.get("training_mode", 0))
    if training_mode != 0:
        raise UnsupportedOpError("BatchNormalization supports training_mode=0 only")
    spatial = int(node.attrs.get("spatial", 1))
    if spatial != 1:
        raise UnsupportedOpError("BatchNormalization supports spatial=1 only")
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "BatchNormalization expects input rank of at least 2"
        )
    channels = input_shape[1]
    for name in node.inputs[1:]:
        shape = _value_shape(graph, name, node)
        if shape != (channels,):
            raise ShapeInferenceError(
                "BatchNormalization parameter shape must be "
                f"({channels},), got {shape}"
            )
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            "BatchNormalization output shape must match input shape, "
            f"got {output_shape}"
        )
    return _BatchNormSpec(
        shape=input_shape,
        channels=channels,
        epsilon=epsilon,
    )


@register_lowering("BatchNormalization")
def lower_batch_normalization(graph: Graph, node: Node) -> BatchNormOp:
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if op_dtype not in {"float", "double", "float16"}:
        raise UnsupportedOpError(
            "BatchNormalization supports float16, float, and double inputs only"
        )
    spec = _resolve_batch_norm_spec(graph, node)
    return BatchNormOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        bias=node.inputs[2],
        mean=node.inputs[3],
        variance=node.inputs[4],
        output=node.outputs[0],
        shape=spec.shape,
        channels=spec.channels,
        epsilon=spec.epsilon,
        dtype=op_dtype,
    )
