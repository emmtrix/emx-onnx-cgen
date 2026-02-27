from __future__ import annotations

from ..ir.ops import LogSoftmaxOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import onnx_opset_version
from .registry import register_lowering


@register_lowering("LogSoftmax")
def lower_logsoftmax(graph: Graph, node: Node) -> LogSoftmaxOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LogSoftmax must have 1 input and 1 output")
    opset_version = onnx_opset_version(graph)
    return LogSoftmaxOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        axis=int(node.attrs["axis"]) if "axis" in node.attrs else None,
        use_legacy_axis_semantics=opset_version is not None and opset_version < 13,
    )
