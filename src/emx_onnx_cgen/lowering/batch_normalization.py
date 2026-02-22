from __future__ import annotations

from dataclasses import dataclass

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import BatchNormOp
from .common import optional_name
from .registry import register_lowering


@dataclass(frozen=True)
class _BatchNormSpec:
    shape: tuple[int, ...]
    channels: int
    epsilon: float
    momentum: float
    training_mode: bool
    running_mean: str | None
    running_variance: str | None


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
    if len(node.inputs) != 5:
        raise UnsupportedOpError("BatchNormalization must have exactly 5 inputs")
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise UnsupportedOpError("BatchNormalization must have between 1 and 3 outputs")
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
    if training_mode not in {0, 1}:
        raise UnsupportedOpError("BatchNormalization training_mode must be 0 or 1")
    if training_mode == 0 and len(node.outputs) != 1:
        raise UnsupportedOpError(
            "BatchNormalization requires exactly 1 output when training_mode=0"
        )
    if training_mode == 1 and len(node.outputs) != 3:
        raise UnsupportedOpError(
            "BatchNormalization requires exactly 3 outputs when training_mode=1"
        )
    spatial = int(node.attrs.get("spatial", 1))
    if spatial != 1:
        raise UnsupportedOpError("BatchNormalization supports spatial=1 only")
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    momentum = float(node.attrs.get("momentum", 0.9))

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

    running_mean = optional_name(node.outputs, 1)
    running_variance = optional_name(node.outputs, 2)
    if training_mode == 1:
        assert running_mean is not None
        assert running_variance is not None
        mean_shape = _value_shape(graph, running_mean, node)
        variance_shape = _value_shape(graph, running_variance, node)
        if mean_shape != (channels,):
            raise ShapeInferenceError(
                "BatchNormalization running_mean shape must be "
                f"({channels},), got {mean_shape}"
            )
        if variance_shape != (channels,):
            raise ShapeInferenceError(
                "BatchNormalization running_var shape must be "
                f"({channels},), got {variance_shape}"
            )

    return _BatchNormSpec(
        shape=input_shape,
        channels=channels,
        epsilon=epsilon,
        momentum=momentum,
        training_mode=training_mode == 1,
        running_mean=running_mean,
        running_variance=running_variance,
    )


@register_lowering("BatchNormalization")
def lower_batch_normalization(graph: Graph, node: Node) -> BatchNormOp:
    spec = _resolve_batch_norm_spec(graph, node)
    output_names = [node.outputs[0]]
    if spec.running_mean is not None:
        output_names.append(spec.running_mean)
    if spec.running_variance is not None:
        output_names.append(spec.running_variance)
    op_dtype = _node_dtype(graph, node, *node.inputs, *output_names)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "BatchNormalization supports float16, float, and double inputs only"
        )
    return BatchNormOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        bias=node.inputs[2],
        mean=node.inputs[3],
        variance=node.inputs[4],
        output=node.outputs[0],
        running_mean=spec.running_mean,
        running_variance=spec.running_variance,
        shape=spec.shape,
        channels=spec.channels,
        epsilon=spec.epsilon,
        momentum=spec.momentum,
        training_mode=spec.training_mode,
        dtype=op_dtype,
    )
