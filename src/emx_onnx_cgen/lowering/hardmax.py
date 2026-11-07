from __future__ import annotations

from ..ir.ops import HardmaxOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import onnx_opset_version
from .registry import register_lowering


@register_lowering("Hardmax")
def lower_hardmax(graph: Graph, node: Node) -> HardmaxOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Hardmax must have 1 input and 1 output")
    opset_version = onnx_opset_version(graph)
    return HardmaxOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        axis=int(node.attrs["axis"]) if "axis" in node.attrs else None,
        use_legacy_axis_semantics=opset_version is not None and opset_version < 13,
    )
