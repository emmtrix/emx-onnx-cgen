"""Short-circuit graph inputs with graph outputs into persistent state buffers.

A state pair ``past -> present`` (e.g. a KV cache) declares that the value of
graph output ``present`` becomes graph input ``past`` of the next call. The
generated code then keeps both in ONE global buffer instead of copying the
cache in and out on every call.

The pairs are always explicit (``--state-pair`` or the model metadata key
:data:`STATE_PAIRS_METADATA_KEY`); they are never inferred from names.

Sharing the buffer is only legal when no op reads the old value after the new
value has been written. :func:`apply_state_aliasing` checks that on the
lowered op order and rejects anything else; it never falls back to a copy.
See ``docs/state-aliasing.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

import onnx
import sympy

from shared.scalar_types import ScalarType

from .codegen.c_emitter import LoweredModel
from .dim_params import parse_dim_param
from .errors import CodegenError
from .ir.model import TensorType
from .ir.ops import ConcatOp, ExpandOp, MatMulOp, ReshapeOp, ShapeOp, TransposeOp
from .ir.state import StateBinding, StatePair

STATE_PAIRS_METADATA_KEY = "emx_onnx_cgen.state_pairs"


class StateAliasingError(CodegenError):
    """Raised when a state pair cannot share one buffer."""


def parse_state_pairs(values: Sequence[str]) -> tuple[StatePair, ...]:
    """Parse ``INPUT=OUTPUT`` entries; each value may hold comma-separated pairs."""
    pairs: list[StatePair] = []
    for value in values:
        for entry in value.split(","):
            entry = entry.strip()
            if not entry:
                continue
            input_name, separator, output_name = entry.partition("=")
            input_name = input_name.strip()
            output_name = output_name.strip()
            if not separator or not input_name or not output_name:
                raise ValueError(
                    f"Invalid state pair {entry!r}; expected INPUT=OUTPUT "
                    "(e.g. past_k_0=new_k_0)"
                )
            pairs.append(StatePair(input_name, output_name))
    _check_unique(pairs)
    return tuple(pairs)


def state_pairs_from_metadata(model: onnx.ModelProto) -> tuple[StatePair, ...]:
    for prop in model.metadata_props:
        if prop.key == STATE_PAIRS_METADATA_KEY:
            try:
                return parse_state_pairs([prop.value])
            except ValueError as exc:
                raise StateAliasingError(
                    f"Invalid model metadata {STATE_PAIRS_METADATA_KEY!r}: {exc}"
                ) from exc
    return ()


def resolve_state_pairs(
    requested: Sequence[StatePair] | None, model: onnx.ModelProto
) -> tuple[StatePair, ...]:
    """Return ``requested`` if given (``()`` disables state), else the metadata pairs."""
    if requested is not None:
        return tuple(requested)
    return state_pairs_from_metadata(model)


def parse_state_pair_args(
    values: Sequence[str] | None, *, disabled: bool = False
) -> tuple[StatePair, ...] | None:
    """Map CLI arguments to :func:`resolve_state_pairs`' ``requested``."""
    if disabled:
        return ()
    if not values:
        return None
    try:
        return parse_state_pairs(values)
    except ValueError as exc:
        raise StateAliasingError(str(exc)) from exc


def _check_unique(pairs: Sequence[StatePair]) -> None:
    inputs = [pair.input_name for pair in pairs]
    outputs = [pair.output_name for pair in pairs]
    for kind, names in (("input", inputs), ("output", outputs)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"State pair {kind} {duplicates[0]!r} is used by more than one pair"
            )


def apply_state_aliasing(
    model: LoweredModel,
    pairs: Sequence[StatePair],
    upper_bounds: Mapping[str, int] | None = None,
) -> LoweredModel:
    """Validate ``pairs`` and let each output share the buffer of its input.

    Every op that writes or reads the output is rewritten to use the input
    name, so the emitter sees one value per state buffer. A growing pair
    (``present = Concat(past, new)``, see :func:`_growing_binding`) keeps both
    names; its Concat only writes the new rows.
    """
    if not pairs:
        return model
    try:
        _check_unique(pairs)
    except ValueError as exc:
        raise StateAliasingError(str(exc)) from exc
    bindings = tuple(_bind_pair(model, pair, upper_bounds or {}) for pair in pairs)
    subgraph_reads = tuple(
        _subgraph_references(info.attrs.values()) for info in model.node_infos
    )
    growing_writers: set[int] = set()
    for binding in bindings:
        if binding.is_growing:
            growing_writers.add(_check_growing_legality(model, binding))
        else:
            _check_alias_legality(model, binding, subgraph_reads)
    rename = {
        binding.output_name: binding.input_name
        for binding in bindings
        if not binding.is_growing
    }
    ops = tuple(
        (
            dataclasses.replace(op, in_place_prefix=True)
            if index in growing_writers
            else op
        ).remap_names(rename)
        for index, op in enumerate(model.ops)
    )
    for source_op, target_op in zip(model.ops, ops):
        model.op_context.copy_derived(source_op, target_op)
    # A state that is passed through unchanged (Identity past -> present)
    # needs no code once both names are one buffer.
    kept = [index for index, op in enumerate(ops) if not op.elides_in_place()]
    return dataclasses.replace(
        model,
        ops=tuple(ops[index] for index in kept),
        node_infos=tuple(model.node_infos[index] for index in kept),
        state_bindings=bindings,
    )


def _bind_pair(
    model: LoweredModel, pair: StatePair, upper_bounds: Mapping[str, int]
) -> StateBinding:
    label = f"State pair {pair.format()}"
    if pair.input_name not in model.input_names:
        raise StateAliasingError(
            f"{label}: {pair.input_name!r} is not a graph input "
            f"(graph inputs: {', '.join(model.input_names) or 'none'})"
        )
    if pair.output_name not in model.output_names:
        raise StateAliasingError(
            f"{label}: {pair.output_name!r} is not a graph output "
            f"(graph outputs: {', '.join(model.output_names) or 'none'})"
        )
    if pair.input_name in model.output_names:
        raise StateAliasingError(
            f"{label}: input {pair.input_name!r} is also a graph output"
        )
    input_index = model.input_names.index(pair.input_name)
    output_index = model.output_names.index(pair.output_name)
    input_type = model.input_types[input_index]
    output_type = model.output_types[output_index]
    for role, name, value_type in (
        ("input", pair.input_name, input_type),
        ("output", pair.output_name, output_type),
    ):
        if not isinstance(value_type, TensorType):
            raise StateAliasingError(
                f"{label}: {role} {name!r} must be a tensor, got a sequence"
            )
        if value_type.is_optional:
            raise StateAliasingError(f"{label}: {role} {name!r} must not be optional")
    input_shape = model.input_shapes[input_index]
    output_shape = model.output_shapes[output_index]
    input_dtype = model.input_dtypes[input_index]
    output_dtype = model.output_dtypes[output_index]
    if input_dtype != output_dtype:
        raise StateAliasingError(
            f"{label}: dtype mismatch, input {pair.input_name!r} is "
            f"{input_dtype.onnx_name} but output {pair.output_name!r} is "
            f"{output_dtype.onnx_name}. Hint: both sides of a state pair must "
            "have the same dtype; insert a Cast before the output."
        )
    # The lowered output types carry no dim_params; the graph values do.
    input_params = _dim_params(model, pair.input_name, len(input_shape))
    output_params = _dim_params(model, pair.output_name, len(output_shape))
    if input_shape != output_shape or input_params != output_params:
        growing = _growing_binding(
            label,
            pair,
            input_shape,
            output_shape,
            input_params,
            output_params,
            input_dtype,
            upper_bounds,
        )
        if growing is not None:
            return growing
    if input_shape != output_shape:
        raise StateAliasingError(
            f"{label}: shape mismatch, input {pair.input_name!r} has shape "
            f"{list(input_shape)} but output {pair.output_name!r} has shape "
            f"{list(output_shape)}. Hint: a state buffer has a fixed size; "
            "export a fixed-size cache (e.g. written with TensorScatter) "
            "instead of a growing one (Concat)."
        )
    if any(dim < 0 for dim in input_shape) or any(input_type.dim_params):
        raise StateAliasingError(
            f"{label}: shape {list(input_shape)} must be static. "
            "Hint: pin dynamic dimensions with --input-dim or export with "
            "static shapes."
        )
    if input_dtype == ScalarType.STRING:
        raise StateAliasingError(f"{label}: string tensors are not supported")
    return StateBinding(
        input_name=pair.input_name,
        output_name=pair.output_name,
        shape=tuple(input_shape),
        dtype=input_dtype,
    )


def _growing_binding(
    label: str,
    pair: StatePair,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    input_params: Sequence[str | None],
    output_params: Sequence[str | None],
    dtype: ScalarType,
    upper_bounds: Mapping[str, int],
) -> StateBinding | None:
    """A pair whose output is the input grown along one dynamic axis.

    ``past[..., n, ...] -> present[..., n + k, ...]`` (a KV cache appended
    with Concat) shares one buffer whose capacity on that axis is the upper
    bound of ``n`` plus ``k``; the input is its first ``n`` rows, the output
    its first ``n + k``. Returns ``None`` if the shapes do not have this form.
    """

    if len(input_shape) != len(output_shape):
        return None
    differing = [
        axis
        for axis in range(len(input_shape))
        if input_shape[axis] != output_shape[axis]
        or input_params[axis] != output_params[axis]
    ]
    if len(differing) != 1:
        return None
    axis = differing[0]
    input_param = input_params[axis]
    output_param = output_params[axis]
    if not input_param or not output_param:
        return None
    length = parse_dim_param(input_param)
    grown = parse_dim_param(output_param)
    if not isinstance(length, sympy.Symbol) or grown is None:
        return None
    growth = sympy.expand(grown - length)
    if not growth.is_Integer or int(growth) < 1:
        return None
    for other in range(len(input_shape)):
        if other != axis and (input_shape[other] < 0 or input_params[other]):
            raise StateAliasingError(
                f"{label}: a growing state may only be dynamic along the growing "
                f"axis {axis}, but axis {other} of {pair.input_name!r} is dynamic "
                f"({list(input_shape)}). Hint: pin it with --input-dim."
            )
    bound = upper_bounds.get(length.name)
    if bound is None:
        raise StateAliasingError(
            f"{label}: {pair.output_name!r} grows {pair.input_name!r} along axis "
            f"{axis} ({input_param} -> {output_param}); the state buffer needs a "
            f"capacity. Hint: pass --input-dim '{length.name}<=N'."
        )
    if dtype == ScalarType.STRING:
        raise StateAliasingError(f"{label}: string tensors are not supported")
    storage = list(input_shape)
    storage[axis] = bound + int(growth)
    return StateBinding(
        input_name=pair.input_name,
        output_name=pair.output_name,
        shape=tuple(storage),
        dtype=dtype,
        growth_axis=axis,
        growth=int(growth),
    )


def _check_growing_legality(model: LoweredModel, binding: StateBinding) -> int:
    """Index of the Concat that appends to a growing state; reject others.

    The output must be ``Concat(input, new..., axis=growth_axis)`` and nothing
    else may write it. The input stays readable everywhere: appending never
    changes its rows.
    """

    label = f"State pair {binding.input_name}={binding.output_name}"
    writers = [
        index
        for index, op in enumerate(model.ops)
        if binding.output_name in op.output_names
    ]
    if len(writers) != 1:
        raise StateAliasingError(
            f"{label}: output {binding.output_name!r} must be written by exactly "
            f"one node, found {len(writers)}"
        )
    writer_index = writers[0]
    writer = model.ops[writer_index]
    rank = len(binding.shape)
    if not (
        isinstance(writer, ConcatOp)
        and writer.inputs
        and writer.inputs[0] == binding.input_name
        and binding.input_name not in writer.inputs[1:]
        and writer.axis % rank == binding.growth_axis
    ):
        raise StateAliasingError(
            f"{label}: {_describe_node(model, writer_index)} writes "
            f"{binding.output_name!r}, but a growing state must be "
            f"Concat({binding.input_name}, ..., axis={binding.growth_axis}) so "
            "that only the new rows are written. Hint: export the cache update "
            "as such a Concat, or pin the length with --input-dim."
        )
    for index, op in enumerate(model.ops):
        if index != writer_index and binding.input_name in op.output_names:
            raise StateAliasingError(
                f"{label}: {_describe_node(model, index)} writes the state input "
                f"{binding.input_name!r}"
            )
        reads_state = {binding.input_name, binding.output_name} & set(op.input_names)
        if (
            index != writer_index
            and reads_state
            and not _reads_capacity_view(model, op)
        ):
            raise StateAliasingError(
                f"{label}: {_describe_node(model, index)} reads "
                f"{', '.join(sorted(reads_state))}, which lives in a buffer of "
                f"capacity {binding.shape[binding.growth_axis]} along axis "
                f"{binding.growth_axis}; its kernel cannot read such a strided "
                "view yet. Hint: remove the pair or pin the length with --input-dim."
            )
    return writer_index


# Kernels that index their inputs with one subscript per axis, so a growing
# state (declared with its capacity, read up to its length) is safe to read.
# ReshapeOp only qualifies when it adds or drops size-1 axes (its generic
# kernel reads through a flat view with dense strides).
_CAPACITY_VIEW_READERS = (ShapeOp, TransposeOp, ReshapeOp, ExpandOp, MatMulOp)


def _reads_capacity_view(model: LoweredModel, op: object) -> bool:
    if not isinstance(op, _CAPACITY_VIEW_READERS):
        return False
    if not isinstance(op, ReshapeOp):
        return True
    input_dims = _named_dims(model, op.input0)
    output_dims = _named_dims(model, op.output)
    loop_vars = tuple(f"i{axis}" for axis in range(len(output_dims)))
    return (
        ReshapeOp.unit_axes_input_index(input_dims, output_dims, loop_vars) is not None
    )


def _dim_params(model: LoweredModel, name: str, rank: int) -> tuple[str | None, ...]:
    """The dim_params of graph value ``name``, padded to ``rank``."""
    try:
        value_type = model.op_context.graph.find_value(name).type
    except KeyError:
        return (None,) * rank
    params = tuple(getattr(value_type, "dim_params", ()) or ())
    if len(params) != rank:
        return (None,) * rank
    return params


def _named_dims(model: LoweredModel, name: str) -> tuple[int | str, ...]:
    shape = model.op_context.shape(name)
    params = _dim_params(model, name, len(shape))
    return tuple(
        params[axis] or f"?{axis}" if dim < 0 else dim for axis, dim in enumerate(shape)
    )


def _check_alias_legality(
    model: LoweredModel,
    binding: StateBinding,
    subgraph_reads: Sequence[set[str]],
) -> None:
    """Reject a pair if the old state is read after the new state is written.

    With one buffer, the first write of the output overwrites the input. The
    writing op itself may read the input only through input fields it
    declares in-place safe (:meth:`OpBase.in_place_input_fields`), e.g. a
    same-index elementwise update or TensorScatter's ``past_cache``.
    """
    label = f"State pair {binding.input_name}={binding.output_name}"
    writer_index = next(
        (
            index
            for index, op in enumerate(model.ops)
            if binding.output_name in op.output_names
        ),
        None,
    )
    if writer_index is None:
        raise StateAliasingError(
            f"{label}: output {binding.output_name!r} is not computed by any "
            "node, so it cannot share a buffer with the input"
        )
    if not _writer_reads_in_place(
        model, writer_index, binding, subgraph_reads[writer_index]
    ):
        raise StateAliasingError(
            f"{label}: {_describe_node(model, writer_index)} reads "
            f"{binding.input_name!r} while writing {binding.output_name!r}, "
            "and its kernel cannot update that buffer in place. "
            "Hint: only same-index elementwise ops and TensorScatter "
            "(past_cache -> output) may read and write a state buffer in one "
            "node; otherwise remove the pair to keep separate buffers."
        )
    for index in range(writer_index + 1, len(model.ops)):
        if (
            binding.input_name in model.ops[index].input_names
            or binding.input_name in subgraph_reads[index]
        ):
            raise StateAliasingError(
                f"{label}: {_describe_node(model, index)} reads "
                f"{binding.input_name!r} after {_describe_node(model, writer_index)} "
                f"has written {binding.output_name!r}; with one shared buffer it "
                "would see the new value. Hint: read the updated state "
                f"{binding.output_name!r} instead (e.g. let Attention consume "
                "the TensorScatter output), or remove the pair to keep separate "
                "input/output buffers."
            )


def _writer_reads_in_place(
    model: LoweredModel,
    writer_index: int,
    binding: StateBinding,
    subgraph_reads: set[str],
) -> bool:
    """Whether every read of the state input by its writer is in-place safe."""
    if binding.input_name in subgraph_reads:
        return False
    writer = model.ops[writer_index]
    reading_fields = writer.input_fields_reading(binding.input_name)
    if reading_fields is None:
        return False
    safe_fields = writer.in_place_input_fields(model.op_context, binding.output_name)
    return set(reading_fields) <= set(safe_fields)


def _subgraph_references(values: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for value in values:
        graphs = value if isinstance(value, (list, tuple)) else (value,)
        for graph in graphs:
            if not isinstance(graph, onnx.GraphProto):
                continue
            for node in graph.node:
                names.update(node.input)
                names.update(
                    _subgraph_references(
                        [attr.g for attr in node.attribute if attr.HasField("g")]
                        + [list(attr.graphs) for attr in node.attribute]
                    )
                )
    return names


def _describe_node(model: LoweredModel, index: int) -> str:
    info = model.node_infos[index]
    name = f" '{info.name}'" if info.name else ""
    return f"node {index} ({info.op_type}{name})"
