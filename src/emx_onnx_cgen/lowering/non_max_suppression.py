from __future__ import annotations

from ..ir.ops import NonMaxSuppressionOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import optional_name
from .registry import register_lowering


@register_lowering("NonMaxSuppression")
def lower_non_max_suppression(graph: Graph, node: Node) -> NonMaxSuppressionOp:
    if node.op_type != "NonMaxSuppression":
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 output, got {len(node.outputs)}"
        )
    if len(node.inputs) < 2 or len(node.inputs) > 5:
        raise UnsupportedOpError(
            f"{node.op_type} must have 2 to 5 inputs, got {len(node.inputs)}"
        )

    boxes = node.inputs[0]
    scores = node.inputs[1]
    max_output_boxes_per_class = optional_name(node.inputs, 2)
    iou_threshold = optional_name(node.inputs, 3)
    score_threshold = optional_name(node.inputs, 4)
    output = node.outputs[0]

    center_point_box = int(node.attrs.get("center_point_box", 0))

    return NonMaxSuppressionOp(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        output=output,
        center_point_box=center_point_box,
    )
