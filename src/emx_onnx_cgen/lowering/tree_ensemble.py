from __future__ import annotations

import math

import onnx
from onnx import numpy_helper

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import TreeEnsembleOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_MODES = {0, 1, 2, 3, 4, 5, 6}


def _as_int_tuple(values: object, *, name: str, node: Node) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise UnsupportedOpError(
            f"{node.op_type} {name} must be an integer list"
        ) from exc


def _as_float_tensor(values: object, *, name: str, node: Node) -> tuple[float, ...]:
    if not isinstance(values, onnx.TensorProto):
        raise UnsupportedOpError(f"{node.op_type} {name} must be a tensor")
    array = numpy_helper.to_array(values)
    return tuple(float(value) for value in array.reshape(-1))


def _as_mode_tensor(values: object, *, node: Node) -> tuple[int, ...]:
    if not isinstance(values, onnx.TensorProto):
        raise UnsupportedOpError(f"{node.op_type} nodes_modes must be a tensor")
    array = numpy_helper.to_array(values)
    modes = tuple(int(value) for value in array.reshape(-1))
    unsupported = sorted(set(modes).difference(_SUPPORTED_MODES))
    if unsupported:
        raise UnsupportedOpError(
            f"{node.op_type} unsupported nodes_modes values: {unsupported}"
        )
    return modes


def _normalize_node_splits(
    *, node_modes: tuple[int, ...], node_splits: tuple[float, ...], node: Node
) -> tuple[float, ...]:
    normalized: list[float] = []
    for mode, split in zip(node_modes, node_splits, strict=True):
        if mode == 6:
            normalized.append(0.0 if math.isnan(split) else split)
            continue
        if math.isnan(split):
            raise UnsupportedOpError(
                f"{node.op_type} nodes_splits contains NaN for non BRANCH_MEMBER node"
            )
        normalized.append(split)
    return tuple(normalized)


def _build_membership_values(
    *, node_modes: tuple[int, ...], membership_values: tuple[float, ...] | None
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    starts: list[int] = []
    ends: list[int] = []
    compact_values: list[float] = []
    cursor = 0
    values = membership_values or ()
    for mode in node_modes:
        if mode != 6:
            starts.append(0)
            ends.append(0)
            continue
        start = len(compact_values)
        while cursor < len(values) and not math.isnan(values[cursor]):
            compact_values.append(values[cursor])
            cursor += 1
        if cursor < len(values) and math.isnan(values[cursor]):
            cursor += 1
        starts.append(start)
        ends.append(len(compact_values))
    return tuple(compact_values), tuple(starts), tuple(ends)


@register_lowering("TreeEnsemble")
def lower_tree_ensemble(graph: Graph, node: Node) -> TreeEnsembleOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} expects 1 input and 1 output")
    input0, output = node.inputs[0], node.outputs[0]
    input_shape = value_shape(graph, input0, node)
    output_shape = value_shape(graph, output, node)
    input_dtype = value_dtype(graph, input0, node)
    output_dtype = value_dtype(graph, output, node)

    if len(input_shape) != 2:
        raise UnsupportedOpError(f"{node.op_type} input rank must be 2")
    if len(output_shape) != 2 or output_shape[0] != input_shape[0]:
        raise UnsupportedOpError(
            f"{node.op_type} output shape must be ({input_shape[0]}, n_targets)"
        )
    if input_dtype not in {ScalarType.F32, ScalarType.F64}:
        raise UnsupportedOpError(
            f"{node.op_type} input dtype must be float32 or float64"
        )
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} output dtype must match input dtype ({input_dtype.name.lower()})"
        )

    required = (
        "leaf_targetids",
        "leaf_weights",
        "nodes_falseleafs",
        "nodes_falsenodeids",
        "nodes_featureids",
        "nodes_modes",
        "nodes_splits",
        "nodes_trueleafs",
        "nodes_truenodeids",
        "tree_roots",
    )
    attrs: dict[str, object] = {}
    for name in required:
        value = node.attrs.get(name)
        if value is None:
            raise UnsupportedOpError(f"{node.op_type} requires {name} attribute")
        attrs[name] = value

    aggregate_function = int(node.attrs.get("aggregate_function", 1))
    if aggregate_function != 1:
        raise UnsupportedOpError(
            f"{node.op_type} only aggregate_function=SUM (1) is supported"
        )
    post_transform = int(node.attrs.get("post_transform", 0))
    if post_transform != 0:
        raise UnsupportedOpError(
            f"{node.op_type} only post_transform=NONE (0) is supported"
        )

    n_targets = int(node.attrs.get("n_targets", output_shape[1]))
    if n_targets != output_shape[1]:
        raise UnsupportedOpError(
            f"{node.op_type} n_targets must match output shape second dim"
        )

    node_modes = _as_mode_tensor(attrs["nodes_modes"], node=node)
    node_feature_ids = _as_int_tuple(
        attrs["nodes_featureids"], name="nodes_featureids", node=node
    )
    node_splits_raw = _as_float_tensor(
        attrs["nodes_splits"], name="nodes_splits", node=node
    )
    node_true_leafs = _as_int_tuple(
        attrs["nodes_trueleafs"], name="nodes_trueleafs", node=node
    )
    node_true_ids = _as_int_tuple(
        attrs["nodes_truenodeids"], name="nodes_truenodeids", node=node
    )
    node_false_leafs = _as_int_tuple(
        attrs["nodes_falseleafs"], name="nodes_falseleafs", node=node
    )
    node_false_ids = _as_int_tuple(
        attrs["nodes_falsenodeids"], name="nodes_falsenodeids", node=node
    )

    node_count = len(node_modes)
    node_fields = (
        node_feature_ids,
        node_splits_raw,
        node_true_leafs,
        node_true_ids,
        node_false_leafs,
        node_false_ids,
    )
    if any(len(field) != node_count for field in node_fields):
        raise UnsupportedOpError(f"{node.op_type} all nodes_* attributes must align")

    node_splits = _normalize_node_splits(
        node_modes=node_modes,
        node_splits=node_splits_raw,
        node=node,
    )

    leaf_target_ids = _as_int_tuple(
        attrs["leaf_targetids"], name="leaf_targetids", node=node
    )
    leaf_weights = _as_float_tensor(
        attrs["leaf_weights"], name="leaf_weights", node=node
    )
    if len(leaf_target_ids) != len(leaf_weights):
        raise UnsupportedOpError(
            f"{node.op_type} leaf_targetids and leaf_weights must have same length"
        )

    tree_roots = _as_int_tuple(attrs["tree_roots"], name="tree_roots", node=node)

    membership_values_attr = node.attrs.get("membership_values")
    membership_values_raw = (
        _as_float_tensor(membership_values_attr, name="membership_values", node=node)
        if membership_values_attr is not None
        else None
    )
    membership_values, member_range_starts, member_range_ends = (
        _build_membership_values(
            node_modes=node_modes,
            membership_values=membership_values_raw,
        )
    )

    return TreeEnsembleOp(
        input0=input0,
        output=output,
        aggregate_function=aggregate_function,
        post_transform=post_transform,
        tree_roots=tree_roots,
        node_feature_ids=node_feature_ids,
        node_modes=node_modes,
        node_splits=node_splits,
        node_true_ids=node_true_ids,
        node_true_leafs=node_true_leafs,
        node_false_ids=node_false_ids,
        node_false_leafs=node_false_leafs,
        membership_values=membership_values,
        member_range_starts=member_range_starts,
        member_range_ends=member_range_ends,
        leaf_target_ids=leaf_target_ids,
        leaf_weights=leaf_weights,
        n_targets=n_targets,
    )
