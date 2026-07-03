"""Build a JSON-serializable report of shape inference results.

The report lists every tensor named in the original ONNX model (graph inputs,
graph outputs, and internal node outputs) together with the type declared in
the model file (``model``) and the type computed by the compiler's shape
inference (``inferred``). Only names taken from the original model appear in
the report; tensors introduced by internal transformations (node expansions,
materialized constants) are excluded so the output is stable across compiler
versions and can be used as a reference.
"""

from __future__ import annotations

from itertools import chain
from typing import Iterable

import onnx

from .dtypes import scalar_type_from_onnx
from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import SequenceType, TensorType
from .ir.op_context import OpContext

REPORT_FORMAT = "emx-onnx-cgen-shape-inference"
REPORT_VERSION = 1


def _tensor_type_protos(
    value_info: onnx.ValueInfoProto,
) -> tuple[onnx.TypeProto.Tensor | None, bool]:
    """Return the tensor type proto of ``value_info`` and a sequence flag."""
    kind = value_info.type.WhichOneof("value")
    if kind == "optional_type":
        elem = value_info.type.optional_type.elem_type
        kind = elem.WhichOneof("value")
        if kind == "tensor_type":
            return elem.tensor_type, False
        if kind == "sequence_type":
            seq_elem = elem.sequence_type.elem_type
            if seq_elem.WhichOneof("value") == "tensor_type":
                return seq_elem.tensor_type, True
        return None, False
    if kind == "tensor_type":
        return value_info.type.tensor_type, False
    if kind == "sequence_type":
        elem = value_info.type.sequence_type.elem_type
        if elem.WhichOneof("value") == "tensor_type":
            return elem.tensor_type, True
    return None, False


def _declared_entry(value_info: onnx.ValueInfoProto | None) -> dict[str, object] | None:
    if value_info is None:
        return None
    tensor_type, is_sequence = _tensor_type_protos(value_info)
    if tensor_type is None:
        return None
    entry: dict[str, object] = {}
    if is_sequence:
        entry["sequence"] = True
    if tensor_type.HasField("elem_type") and tensor_type.elem_type != 0:
        scalar = scalar_type_from_onnx(tensor_type.elem_type)
        if scalar is not None:
            entry["dtype"] = scalar.onnx_name
        else:
            entry["dtype"] = onnx.TensorProto.DataType.Name(
                tensor_type.elem_type
            ).lower()
    if tensor_type.HasField("shape"):
        dims: list[object] = []
        for dim in tensor_type.shape.dim:
            which = dim.WhichOneof("value")
            if which == "dim_value":
                dims.append(int(dim.dim_value))
            elif which == "dim_param" and dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        entry["dims"] = dims
    return entry or None


def _concrete_dims(
    shape: Iterable[int],
    dim_params: tuple[str | None, ...],
    known_dim_params: frozenset[str],
) -> list[object]:
    dims: list[object] = []
    for index, dim in enumerate(shape):
        if dim >= 0:
            dims.append(int(dim))
            continue
        # Unresolved dim: report the model's dim_param name if it has one,
        # never the synthetic parameter names the importer invents.
        dim_param = dim_params[index] if index < len(dim_params) else None
        if dim_param and dim_param in known_dim_params:
            dims.append(dim_param)
        else:
            dims.append(None)
    return dims


def _inferred_entry(
    op_context: OpContext,
    name: str,
    known_dim_params: frozenset[str],
) -> dict[str, object] | None:
    try:
        value = op_context.graph.find_value(name)
    except KeyError:
        value = None
    if value is not None and isinstance(value.type, SequenceType):
        elem = value.type.elem
        return {
            "sequence": True,
            "dtype": elem.dtype.onnx_name,
            "dims": _concrete_dims(elem.shape, elem.dim_params, known_dim_params),
        }
    try:
        shape = op_context.shape(name)
        dtype = op_context.dtype(name)
    except (KeyError, ShapeInferenceError, UnsupportedOpError):
        return None
    dim_params: tuple[str | None, ...] = ()
    if isinstance(value.type if value is not None else None, TensorType):
        if len(value.type.dim_params) == len(shape):
            dim_params = value.type.dim_params
    return {
        "dtype": dtype.onnx_name,
        "dims": _concrete_dims(shape, dim_params, known_dim_params),
    }


def _collect_model_dim_params(graph: onnx.GraphProto) -> frozenset[str]:
    params: set[str] = set()
    for value_info in chain(graph.input, graph.output, graph.value_info):
        tensor_type, _is_sequence = _tensor_type_protos(value_info)
        if tensor_type is None or not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if dim.WhichOneof("value") == "dim_param" and dim.dim_param:
                params.add(dim.dim_param)
    return frozenset(params)


def build_shape_inference_report(
    model: onnx.ModelProto,
    op_context: OpContext,
) -> dict[str, object]:
    """Build the report for ``model`` using inference results in ``op_context``.

    ``model`` must be the original (unprepared) model so declared shapes and
    tensor names match the model file; ``op_context`` must come from a lowered
    model so inferred shapes match what code generation uses.
    """
    graph = model.graph
    known_dim_params = _collect_model_dim_params(graph)
    initializer_names = {initializer.name for initializer in graph.initializer}
    declared_by_name: dict[str, onnx.ValueInfoProto] = {}
    for value_info in chain(graph.input, graph.output, graph.value_info):
        declared_by_name.setdefault(value_info.name, value_info)
    output_names = {value_info.name for value_info in graph.output}
    tensors: dict[str, dict[str, object]] = {}

    def add(name: str, kind: str) -> None:
        if not name or name in initializer_names or name in tensors:
            return
        tensors[name] = {
            "kind": kind,
            "model": _declared_entry(declared_by_name.get(name)),
            "inferred": _inferred_entry(op_context, name, known_dim_params),
        }

    for value_info in graph.input:
        add(value_info.name, "input")
    for node in graph.node:
        for output_name in node.output:
            add(output_name, "output" if output_name in output_names else "internal")
    for value_info in graph.output:
        add(value_info.name, "output")

    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "tensors": tensors,
    }
