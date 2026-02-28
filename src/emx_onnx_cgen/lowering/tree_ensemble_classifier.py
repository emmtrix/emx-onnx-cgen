from __future__ import annotations

from typing import Iterable

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import TreeEnsembleClassifierOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


_MODE_IDS = {
    "BRANCH_LEQ": 0,
    "BRANCH_LT": 1,
    "BRANCH_GTE": 2,
    "BRANCH_GT": 3,
    "BRANCH_EQ": 4,
    "BRANCH_NEQ": 5,
    "LEAF": 7,
}


def _as_int_tuple(values: object, *, name: str, node: Node) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise UnsupportedOpError(f"{node.op_type} {name} must be an integer list") from exc


def _as_float_tuple(values: object, *, name: str, node: Node) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise UnsupportedOpError(f"{node.op_type} {name} must be a float list") from exc


def _as_mode_tuple(values: object, *, node: Node) -> tuple[int, ...]:
    mode_ids: list[int] = []
    for raw in values:  # type: ignore[assignment]
        mode = raw.decode() if isinstance(raw, bytes) else str(raw)
        mode_id = _MODE_IDS.get(mode)
        if mode_id is None:
            raise UnsupportedOpError(f"{node.op_type} mode '{mode}' is not supported")
        mode_ids.append(mode_id)
    return tuple(mode_ids)


def _require_attrs(node: Node, names: Iterable[str]) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for name in names:
        value = node.attrs.get(name)
        if value is None:
            raise UnsupportedOpError(f"{node.op_type} requires {name} attribute")
        attrs[name] = value
    return attrs


@register_lowering("TreeEnsembleClassifier")
def lower_tree_ensemble_classifier(graph: Graph, node: Node) -> TreeEnsembleClassifierOp:
    if len(node.inputs) != 1 or len(node.outputs) != 2:
        raise UnsupportedOpError(f"{node.op_type} expects 1 input and 2 outputs")
    input0, label, probabilities = node.inputs[0], node.outputs[0], node.outputs[1]
    input_shape = value_shape(graph, input0, node)
    label_shape = value_shape(graph, label, node)
    probabilities_shape = value_shape(graph, probabilities, node)
    input_dtype = value_dtype(graph, input0, node)
    label_dtype = value_dtype(graph, label, node)
    prob_dtype = value_dtype(graph, probabilities, node)

    if len(input_shape) != 2:
        raise UnsupportedOpError(f"{node.op_type} input rank must be 2")
    if len(label_shape) != 1 or label_shape[0] != input_shape[0]:
        raise UnsupportedOpError(f"{node.op_type} label output shape must be ({input_shape[0]},)")
    if len(probabilities_shape) != 2 or probabilities_shape[0] != input_shape[0]:
        raise UnsupportedOpError(
            f"{node.op_type} probabilities output shape must be ({input_shape[0]}, C)"
        )
    if input_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(f"{node.op_type} input dtype must be float32 or float64")
    if label_dtype != ScalarType.I64:
        raise UnsupportedOpError(f"{node.op_type} label output dtype must be int64")
    if prob_dtype != ScalarType.F32:
        raise UnsupportedOpError(f"{node.op_type} probabilities dtype must be float32")

    attrs = _require_attrs(
        node,
        (
            "classlabels_int64s",
            "nodes_treeids",
            "nodes_nodeids",
            "nodes_featureids",
            "nodes_modes",
            "nodes_values",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "class_treeids",
            "class_nodeids",
            "class_ids",
            "class_weights",
        ),
    )
    class_labels = _as_int_tuple(attrs["classlabels_int64s"], name="classlabels_int64s", node=node)
    if probabilities_shape[1] != len(class_labels):
        raise UnsupportedOpError(
            f"{node.op_type} class count mismatch between classlabels and probabilities output"
        )

    post_transform_raw = node.attrs.get("post_transform", b"NONE")
    post_transform = (
        post_transform_raw.decode()
        if isinstance(post_transform_raw, bytes)
        else str(post_transform_raw)
    )
    if post_transform not in {"NONE", "LOGISTIC"}:
        raise UnsupportedOpError(
            f"{node.op_type} post_transform must be NONE or LOGISTIC, got {post_transform}"
        )

    return TreeEnsembleClassifierOp(
        input0=input0,
        label=label,
        probabilities=probabilities,
        output=probabilities,
        post_transform=post_transform,
        class_labels=class_labels,
        node_tree_ids=_as_int_tuple(attrs["nodes_treeids"], name="nodes_treeids", node=node),
        node_node_ids=_as_int_tuple(attrs["nodes_nodeids"], name="nodes_nodeids", node=node),
        node_feature_ids=_as_int_tuple(attrs["nodes_featureids"], name="nodes_featureids", node=node),
        node_modes=_as_mode_tuple(attrs["nodes_modes"], node=node),
        node_values=_as_float_tuple(attrs["nodes_values"], name="nodes_values", node=node),
        node_true_ids=_as_int_tuple(attrs["nodes_truenodeids"], name="nodes_truenodeids", node=node),
        node_false_ids=_as_int_tuple(attrs["nodes_falsenodeids"], name="nodes_falsenodeids", node=node),
        class_tree_ids=_as_int_tuple(attrs["class_treeids"], name="class_treeids", node=node),
        class_node_ids=_as_int_tuple(attrs["class_nodeids"], name="class_nodeids", node=node),
        class_ids=_as_int_tuple(attrs["class_ids"], name="class_ids", node=node),
        class_weights=_as_float_tuple(attrs["class_weights"], name="class_weights", node=node),
        output_shape=probabilities_shape,
    )
