"""Short-circuit graph inputs with graph outputs into persistent state buffers.

A state pair ``past -> present`` (e.g. a KV cache) declares that the value of
graph output ``present`` becomes graph input ``past`` of the next call. The
generated code then keeps both in ONE buffer that lives in a caller-owned
state struct instead of copying the cache in and out on every call.

The pairs are always explicit (``--state-pair`` or the model metadata key
:data:`STATE_PAIRS_METADATA_KEY`); they are never inferred from names.

Sharing the buffer is only legal when no op reads the old value after the new
value has been written. :func:`apply_state_aliasing` checks that on the
lowered op order and rejects anything else; it never falls back to a copy.
See ``docs/state-aliasing.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import onnx

from shared.scalar_types import ScalarType

from .codegen.c_emitter import LoweredModel
from .errors import CodegenError
from .ir.model import TensorType
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
    model: LoweredModel, pairs: Sequence[StatePair]
) -> LoweredModel:
    """Validate ``pairs`` and let each output share the buffer of its input.

    Every op that writes or reads the output is rewritten to use the input
    name, so the emitter sees one value per state buffer.
    """
    if not pairs:
        return model
    try:
        _check_unique(pairs)
    except ValueError as exc:
        raise StateAliasingError(str(exc)) from exc
    bindings = tuple(_bind_pair(model, pair) for pair in pairs)
    subgraph_reads = tuple(
        _subgraph_references(info.attrs.values()) for info in model.node_infos
    )
    for binding in bindings:
        _check_alias_legality(model, binding, subgraph_reads)
    rename = {binding.output_name: binding.input_name for binding in bindings}
    ops = tuple(op.remap_names(rename) for op in model.ops)
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


def _bind_pair(model: LoweredModel, pair: StatePair) -> StateBinding:
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
