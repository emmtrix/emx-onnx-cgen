from __future__ import annotations

from dataclasses import dataclass

from ..codegen.c_emitter import LrnOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@dataclass(frozen=True)
class LrnSpec:
    shape: tuple[int, ...]
    channels: int
    size: int
    half: int
    alpha: float
    beta: float
    bias: float


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


def resolve_lrn_spec(graph: Graph, node: Node) -> LrnSpec:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LRN must have 1 input and 1 output")
    supported_attrs = {"alpha", "beta", "bias", "size"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("LRN has unsupported attributes")
    size = int(node.attrs.get("size", 0))
    if size <= 0:
        raise UnsupportedOpError("LRN size must be a positive integer")
    if size % 2 == 0:
        raise UnsupportedOpError("LRN size must be odd")
    alpha = float(node.attrs.get("alpha", 0.0001))
    beta = float(node.attrs.get("beta", 0.75))
    bias = float(node.attrs.get("bias", 1.0))
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 2:
        raise UnsupportedOpError("LRN expects input rank of at least 2")
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            "LRN output shape must match input shape, "
            f"got {output_shape} for input {input_shape}"
        )
    return LrnSpec(
        shape=input_shape,
        channels=input_shape[1],
        size=size,
        half=size // 2,
        alpha=alpha,
        beta=beta,
        bias=bias,
    )


@register_lowering("LRN")
def lower_lrn(graph: Graph, node: Node) -> LrnOp:
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if op_dtype != "float":
        raise UnsupportedOpError("LRN supports float inputs only")
    spec = resolve_lrn_spec(graph, node)
    return LrnOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=spec.shape,
        channels=spec.channels,
        size=spec.size,
        half=spec.half,
        alpha=spec.alpha,
        beta=spec.beta,
        bias=spec.bias,
        dtype=op_dtype,
    )
