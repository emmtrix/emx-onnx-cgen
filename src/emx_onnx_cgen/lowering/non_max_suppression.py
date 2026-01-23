from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import NonMaxSuppressionOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import optional_name, shape_product, value_dtype, value_shape
from .registry import register_lowering


def _validate_scalar_input(
    graph: Graph,
    name: str,
    node: Node,
    *,
    allowed_dtypes: set[ScalarType],
    label: str,
) -> tuple[ScalarType, tuple[int, ...]]:
    dtype = value_dtype(graph, name, node)
    if dtype not in allowed_dtypes:
        allowed = ", ".join(sorted(d.onnx_name for d in allowed_dtypes))
        raise UnsupportedOpError(
            f"{node.op_type} {label} must be {allowed}, got {dtype.onnx_name}"
        )
    shape = value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        total = shape_product(shape)
        if total != 1:
            raise ShapeInferenceError(
                f"{node.op_type} {label} must be a scalar tensor, got shape {shape}"
            )
    return dtype, shape


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

    boxes_shape = value_shape(graph, boxes, node)
    scores_shape = value_shape(graph, scores, node)
    if len(boxes_shape) != 3 or boxes_shape[2] != 4:
        raise ShapeInferenceError(
            f"{node.op_type} boxes input must have shape "
            f"[num_batches, num_boxes, 4], got {boxes_shape}"
        )
    if len(scores_shape) != 3:
        raise ShapeInferenceError(
            f"{node.op_type} scores input must have shape "
            f"[num_batches, num_classes, num_boxes], got {scores_shape}"
        )
    if boxes_shape[0] != scores_shape[0]:
        raise ShapeInferenceError(
            f"{node.op_type} boxes/scores batch dims must match, "
            f"got {boxes_shape[0]} and {scores_shape[0]}"
        )
    if boxes_shape[1] != scores_shape[2]:
        raise ShapeInferenceError(
            f"{node.op_type} boxes num_boxes dim {boxes_shape[1]} "
            f"must match scores num_boxes dim {scores_shape[2]}"
        )

    boxes_dtype = value_dtype(graph, boxes, node)
    scores_dtype = value_dtype(graph, scores, node)
    if boxes_dtype != scores_dtype or not boxes_dtype.is_float:
        raise UnsupportedOpError(
            f"{node.op_type} boxes and scores must be the same float dtype, "
            f"got {boxes_dtype.onnx_name} and {scores_dtype.onnx_name}"
        )

    max_output_dtype = None
    max_output_shape = None
    if max_output_boxes_per_class is not None:
        max_output_dtype, max_output_shape = _validate_scalar_input(
            graph,
            max_output_boxes_per_class,
            node,
            allowed_dtypes={ScalarType.I32, ScalarType.I64},
            label="max_output_boxes_per_class input",
        )

    iou_threshold_dtype = None
    iou_threshold_shape = None
    if iou_threshold is not None:
        iou_threshold_dtype, iou_threshold_shape = _validate_scalar_input(
            graph,
            iou_threshold,
            node,
            allowed_dtypes={ScalarType.F32, ScalarType.F64},
            label="iou_threshold input",
        )

    score_threshold_dtype = None
    score_threshold_shape = None
    if score_threshold is not None:
        score_threshold_dtype, score_threshold_shape = _validate_scalar_input(
            graph,
            score_threshold,
            node,
            allowed_dtypes={ScalarType.F32, ScalarType.F64},
            label="score_threshold input",
        )

    output_shape = value_shape(graph, output, node)
    if len(output_shape) != 2 or output_shape[1] != 3:
        raise ShapeInferenceError(
            f"{node.op_type} output must have shape [num_selected, 3], "
            f"got {output_shape}"
        )
    output_dtype = value_dtype(graph, output, node)
    if output_dtype != ScalarType.I64:
        raise UnsupportedOpError(
            f"{node.op_type} output dtype must be int64"
        )

    center_point_box = int(node.attrs.get("center_point_box", 0))
    if center_point_box not in {0, 1}:
        raise UnsupportedOpError(
            f"{node.op_type} center_point_box must be 0 or 1, got {center_point_box}"
        )

    return NonMaxSuppressionOp(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        output=output,
        boxes_shape=boxes_shape,
        scores_shape=scores_shape,
        output_shape=output_shape,
        center_point_box=center_point_box,
        boxes_dtype=boxes_dtype,
        output_dtype=output_dtype,
        max_output_dtype=max_output_dtype,
        max_output_shape=max_output_shape,
        iou_threshold_dtype=iou_threshold_dtype,
        iou_threshold_shape=iou_threshold_shape,
        score_threshold_dtype=score_threshold_dtype,
        score_threshold_shape=score_threshold_shape,
    )
