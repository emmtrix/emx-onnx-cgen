"""Parse and apply CLI overrides that pin dynamic input dimensions.

ONNX models frequently declare inputs with dynamic dimensions, either as named
symbolic parameters (``dim_param`` such as ``batch``) or as fully unknown axes.
The code generator turns such dimensions into runtime parameters of the
generated C function, which makes the buffers C99 variable-length arrays.
Users sometimes want a fully static signature instead -- for example to emit
``void model(const float in[4][3][4], ...)`` rather than
``void model(int batch, const float in[batch][3][4], ...)`` -- either for a
simpler API or to avoid VLAs entirely on toolchains that do not support them
well (MSVC has none, C11 makes them optional, MISRA C forbids them).

This module recovers the list of dynamic input dimensions (for reporting) and
applies user-provided overrides that fix them to concrete values, mutating the
ONNX model in place so the rest of the pipeline sees static shapes.

Override syntax (``--input-dim KEY=VALUE``):

* ``<dim_param>=<int>`` -- fix every axis carrying that symbolic name, e.g.
  ``batch=1``. When the same ``dim_param`` is shared by inputs, outputs and
  intermediate values, all of them are pinned so the graph stays consistent.
* ``<input>:<axis>=<int>`` -- fix a single axis positionally, e.g. ``images:0=1``.
  If that axis happens to carry a ``dim_param``, the parameter is pinned
  graph-wide as well.

Ordering and consequences:

* Pinning runs after the model is loaded but **before shape inference**, so the
  fixed value propagates to dependent intermediate and output shapes. It can
  therefore be used to resolve "tensor 'X' has dynamic dimensions" code
  generation failures -- but only for dimensions *derived from* the pinned
  input dimension, not for shapes computed from runtime tensor values.
* :func:`apply_input_dim_overrides` only validates that an override targets a
  dynamic input dimension; it does not check that the value is consistent with
  the rest of the graph. Contradictory or partial pinning can make shape
  inference or lowering fail later (e.g. pinning two ``Add`` operands to
  incompatible extents). Named ``dim_param``\\ s are always pinned graph-wide and
  stay consistent by construction; the risk is mainly the positional form on
  unnamed axes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import onnx


_PARAM_KEY_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
_POSITION_KEY_RE = re.compile(r"^(?P<name>.+):(?P<axis>0|[1-9][0-9]*)$")
_VALUE_RE = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class DynamicInputDim:
    """A dynamic axis of a model input."""

    input_name: str
    axis: int
    dim_param: str | None

    def format(self) -> str:
        return f"{self.input_name}[{self.axis}]={self.dim_param or '?'}"


@dataclass(frozen=True)
class AppliedInputDim:
    """A dynamic axis that was pinned to a concrete value."""

    input_name: str
    axis: int
    value: int

    def format(self) -> str:
        return f"{self.input_name}[{self.axis}]={self.value}"


@dataclass(frozen=True)
class InputDimOverrides:
    """Parsed ``--input-dim`` overrides."""

    by_param: dict[str, int]
    by_position: dict[tuple[str, int], int]

    def is_empty(self) -> bool:
        return not self.by_param and not self.by_position


def parse_input_dim_overrides(specs: list[str] | tuple[str, ...]) -> InputDimOverrides:
    """Parse ``--input-dim`` specifications into an :class:`InputDimOverrides`."""

    by_param: dict[str, int] = {}
    by_position: dict[tuple[str, int], int] = {}
    for spec in specs:
        key, sep, raw_value = spec.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid --input-dim {spec!r}: expected KEY=VALUE "
                "(for example batch=1 or images:0=1)."
            )
        if not _VALUE_RE.match(raw_value):
            raise ValueError(
                f"Invalid --input-dim {spec!r}: dimension value must be a positive "
                "integer."
            )
        value = int(raw_value)
        position_match = _POSITION_KEY_RE.match(key)
        if position_match:
            name = position_match.group("name")
            axis = int(position_match.group("axis"))
            position = (name, axis)
            if position in by_position:
                raise ValueError(f"Duplicate --input-dim for {name}:{axis}.")
            by_position[position] = value
            continue
        param_match = _PARAM_KEY_RE.match(key)
        if param_match:
            name = param_match.group("name")
            if name in by_param:
                raise ValueError(f"Duplicate --input-dim for {name}.")
            by_param[name] = value
            continue
        raise ValueError(
            f"Invalid --input-dim {spec!r}: key must be a dim parameter name "
            "(batch) or an input position (images:0)."
        )
    return InputDimOverrides(by_param=by_param, by_position=by_position)


def _initializer_names(model: onnx.ModelProto) -> set[str]:
    names = {initializer.name for initializer in model.graph.initializer}
    names.update(
        sparse_init.values.name for sparse_init in model.graph.sparse_initializer
    )
    return names


def collect_dynamic_input_dims(model: onnx.ModelProto) -> list[DynamicInputDim]:
    """Return the dynamic axes of all (non-initializer) tensor inputs."""

    skip = _initializer_names(model)
    dynamic: list[DynamicInputDim] = []
    for value_info in model.graph.input:
        if value_info.name in skip:
            continue
        if value_info.type.WhichOneof("value") != "tensor_type":
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for axis, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value"):
                continue
            dim_param = dim.dim_param if dim.HasField("dim_param") else None
            dynamic.append(
                DynamicInputDim(
                    input_name=value_info.name,
                    axis=axis,
                    dim_param=dim_param or None,
                )
            )
    return dynamic


def _is_dynamic(dim: onnx.TensorShapeProto.Dimension) -> bool:
    return not dim.HasField("dim_value")


def _set_dim(dim: onnx.TensorShapeProto.Dimension, value: int) -> None:
    dim.ClearField("dim_param")
    dim.dim_value = value


def _fix_dim_param(model: onnx.ModelProto, dim_param: str, value: int) -> bool:
    """Pin every dynamic axis carrying ``dim_param`` graph-wide. Returns hit."""

    graph = model.graph
    changed = False
    for value_info in (*graph.input, *graph.output, *graph.value_info):
        if value_info.type.WhichOneof("value") != "tensor_type":
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            if _is_dynamic(dim) and dim.dim_param == dim_param:
                _set_dim(dim, value)
                changed = True
    return changed


def _is_value_info_fully_static(value_info: onnx.ValueInfoProto) -> bool:
    if value_info.type.WhichOneof("value") != "tensor_type":
        return False
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    return all(dim.HasField("dim_value") for dim in tensor_type.shape.dim)


def apply_input_dim_overrides(
    model: onnx.ModelProto, overrides: InputDimOverrides
) -> list[AppliedInputDim]:
    """Pin dynamic input dimensions in ``model`` according to ``overrides``.

    Mutates ``model`` in place and returns the list of axes that were fixed.
    Raises :class:`ValueError` when an override does not match a dynamic input
    dimension (so typos surface instead of being silently ignored).
    """

    if overrides.is_empty():
        return []

    input_by_name = {value_info.name: value_info for value_info in model.graph.input}
    dynamic_dims = collect_dynamic_input_dims(model)
    dynamic_params = {dim.dim_param for dim in dynamic_dims if dim.dim_param}
    dynamic_positions = {(dim.input_name, dim.axis) for dim in dynamic_dims}

    applied: list[AppliedInputDim] = []

    # Positional overrides first so a param attached to the targeted axis is
    # discovered and pinned graph-wide.
    for (name, axis), value in overrides.by_position.items():
        if (name, axis) not in dynamic_positions:
            if name not in input_by_name:
                raise ValueError(
                    f"--input-dim {name}:{axis}: no model input named {name!r}."
                )
            raise ValueError(
                f"--input-dim {name}:{axis}: axis {axis} of input {name!r} is not "
                "a dynamic dimension."
            )
        tensor_type = input_by_name[name].type.tensor_type
        dim = tensor_type.shape.dim[axis]
        dim_param = dim.dim_param if dim.HasField("dim_param") else ""
        if dim_param:
            _fix_dim_param(model, dim_param, value)
        else:
            _set_dim(dim, value)
        applied.append(AppliedInputDim(input_name=name, axis=axis, value=value))

    for dim_param, value in overrides.by_param.items():
        if dim_param not in dynamic_params:
            raise ValueError(
                f"--input-dim {dim_param}={value}: no dynamic input dimension uses "
                f"the parameter {dim_param!r}."
            )
        _fix_dim_param(model, dim_param, value)
        for dim in dynamic_dims:
            if dim.dim_param == dim_param:
                applied.append(
                    AppliedInputDim(
                        input_name=dim.input_name, axis=dim.axis, value=value
                    )
                )

    # Drop now-stale dynamic intermediate shapes so shape inference recomputes
    # them from the freshly concrete inputs.
    graph = model.graph
    kept = [vi for vi in graph.value_info if _is_value_info_fully_static(vi)]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)

    # Stable, de-duplicated ordering for reporting.
    seen: set[tuple[str, int]] = set()
    ordered: list[AppliedInputDim] = []
    for item in applied:
        key = (item.input_name, item.axis)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered
