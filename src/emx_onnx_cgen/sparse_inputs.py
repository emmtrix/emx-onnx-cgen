"""Sparse tensor graph inputs.

A ``sparse_tensor`` graph input crosses the C interface in COO form: a values
array and an indices array whose shared leading extent is the number of stored
entries, passed at runtime like any other dynamic dimension. The ONNX model is
rewritten accordingly before import, so the rest of the compiler only sees
dense tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .errors import UnsupportedOpError

# Operators that accept a sparse tensor, keyed by (domain, op_type), with the
# index of their sparse input. The rewritten node takes the values and indices
# arrays in place of that input.
_SPARSE_CONSUMERS: dict[tuple[str, str], int] = {
    ("com.microsoft", "SparseToDenseMatMul"): 0,
}


@dataclass(frozen=True)
class SparseInputNames:
    values: str
    indices: str
    nnz: str


def sparse_input_names(name: str) -> SparseInputNames:
    return SparseInputNames(
        values=f"{name}_values", indices=f"{name}_indices", nnz=f"{name}_nnz"
    )


def _sparse_input_rank(value_info: onnx.ValueInfoProto) -> int:
    sparse_type = value_info.type.sparse_tensor_type
    if not sparse_type.HasField("shape"):
        raise UnsupportedOpError(
            f"Sparse tensor input '{value_info.name}' has no shape. "
            "Hint: export the model with a static dense shape for sparse inputs."
        )
    dims = sparse_type.shape.dim
    if not all(dim.HasField("dim_value") for dim in dims):
        raise UnsupportedOpError(
            f"Sparse tensor input '{value_info.name}' needs a static dense shape. "
            "Hint: export the model with static shapes."
        )
    return len(dims)


def _graph_names(graph: onnx.GraphProto) -> set[str]:
    names = {value_info.name for value_info in graph.input}
    names.update(value_info.name for value_info in graph.output)
    names.update(value_info.name for value_info in graph.value_info)
    names.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        names.update(node.input)
        names.update(node.output)
    return names


def expand_sparse_inputs(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    """Replace sparse graph inputs by COO values/indices tensor inputs."""
    graph = model.graph
    sparse_inputs = {
        value_info.name: value_info
        for value_info in graph.input
        if value_info.type.WhichOneof("value") == "sparse_tensor_type"
    }
    if not sparse_inputs:
        return model, False
    existing = _graph_names(graph)
    expanded = onnx.ModelProto()
    expanded.CopyFrom(model)
    new_inputs: list[onnx.ValueInfoProto] = []
    for value_info in expanded.graph.input:
        if value_info.name not in sparse_inputs:
            new_inputs.append(value_info)
            continue
        rank = _sparse_input_rank(value_info)
        names = sparse_input_names(value_info.name)
        clashes = sorted({names.values, names.indices, names.nnz} & existing)
        if clashes:
            raise UnsupportedOpError(
                f"Sparse tensor input '{value_info.name}' expands to names that "
                f"are already used: {', '.join(clashes)}. Hint: rename them."
            )
        new_inputs.append(
            helper.make_tensor_value_info(
                names.values,
                value_info.type.sparse_tensor_type.elem_type,
                [names.nnz],
            )
        )
        new_inputs.append(
            helper.make_tensor_value_info(
                names.indices, onnx.TensorProto.INT64, [names.nnz, rank]
            )
        )
    del expanded.graph.input[:]
    expanded.graph.input.extend(new_inputs)
    for value_info in expanded.graph.output:
        if value_info.name in sparse_inputs:
            raise UnsupportedOpError(
                f"Sparse tensor input '{value_info.name}' is also a graph output; "
                "sparse graph outputs are not supported."
            )
    for node in expanded.graph.node:
        if not any(name in sparse_inputs for name in node.input):
            continue
        sparse_index = _SPARSE_CONSUMERS.get((node.domain, node.op_type))
        inputs = list(node.input)
        for index, name in enumerate(inputs):
            if name in sparse_inputs and index != sparse_index:
                raise UnsupportedOpError(
                    f"{node.op_type} (node {node.name or '<unnamed>'}) cannot "
                    f"take the sparse tensor '{name}' as input {index}. "
                    "Hint: densify it before export."
                )
        names = sparse_input_names(inputs[sparse_index])
        del node.input[:]
        node.input.extend(
            [
                *inputs[:sparse_index],
                names.values,
                names.indices,
                *inputs[sparse_index + 1 :],
            ]
        )
    return expanded, True


def coo_arrays(
    sparse: onnx.SparseTensorProto,
) -> tuple[np.ndarray, np.ndarray]:
    """Values and ``[nnz, rank]`` coordinates of a sparse tensor proto."""
    dims = tuple(int(dim) for dim in sparse.dims)
    values = numpy_helper.to_array(sparse.values).reshape(-1)
    if sparse.indices.dims:
        indices = numpy_helper.to_array(sparse.indices).astype(np.int64)
    else:
        indices = np.zeros((0, len(dims)), dtype=np.int64)
    if indices.ndim == 1:
        # Linearized COO indices address the dense tensor in row-major order.
        indices = np.stack(np.unravel_index(indices, dims), axis=1).astype(np.int64)
    indices = indices.reshape(values.shape[0], len(dims))
    return values, indices
