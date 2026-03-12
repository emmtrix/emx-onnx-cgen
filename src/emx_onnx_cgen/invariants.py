"""Central invariant checks for the compiler pipeline (fail-fast gates).

Each function validates a set of structural invariants at a specific
pipeline boundary.  Gate checks raise ``ShapeInferenceError`` or
``UnsupportedOpError`` immediately when an invariant is violated so
that problems surface close to their origin rather than manifesting
as hard-to-diagnose numerical errors in later stages.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, TensorType
from .ir.op_base import OpBase
from .ir.op_context import OpContext


def check_graph_integrity(graph: Graph) -> None:
    """Import-Gate: validate that the imported graph is structurally complete.

    Checks performed:
    - Graph has at least one output.
    - Graph has at least one node.
    - All tensor values referenced by the graph have either non-negative
      shape dimensions or explicitly marked dynamic dimensions via
      ``dim_params``.
    """
    if not graph.outputs:
        raise UnsupportedOpError("Graph must have at least one output")
    if not graph.nodes:
        raise UnsupportedOpError("Graph must contain at least one node")

    for value in graph.inputs + graph.outputs + graph.values:
        if not isinstance(value.type, TensorType):
            continue
        for dim_index, dim in enumerate(value.type.shape):
            dim_param = (
                value.type.dim_params[dim_index]
                if dim_index < len(value.type.dim_params)
                else None
            )
            if dim < 0 and not dim_param:
                raise ShapeInferenceError(
                    f"Negative dimension {dim} at index {dim_index} "
                    f"for value '{value.name}'. "
                    "Hint: export with static shapes or provide "
                    "--shape-inference-shapes."
                )


def check_lowered_ops(ops: Sequence[OpBase]) -> None:
    """Lowering-Gate: validate that every lowered op is structurally complete.

    Each op must expose at least one output name so that subsequent
    validation and codegen phases can query shapes and dtypes.
    """
    for op in ops:
        outputs = op.output_names
        if not outputs:
            raise UnsupportedOpError(f"Lowered op {op.kind} has no output names")
        for name in outputs:
            if name is not None and not name:
                raise UnsupportedOpError(
                    f"Lowered op {op.kind} has an empty output name"
                )


def check_inferred_types(ops: Sequence[OpBase], ctx: OpContext) -> None:
    """Type-Gate: validate that output dtypes are resolvable after infer_types.

    Every tensor output must have a valid dtype.  Non-tensor outputs
    (e.g. sequences) are skipped.

    ``ctx.dtype()`` raises ``ShapeInferenceError`` for missing tensor
    dtypes — that error propagates uncaught as a genuine gate failure.
    ``UnsupportedOpError`` is caught and ignored because it signals a
    non-tensor output that this gate does not cover.
    """
    for op in ops:
        name = op.primary_output_name
        if name is None:
            continue
        try:
            ctx.dtype(name)
        except UnsupportedOpError:
            # Non-tensor output (e.g. sequence) — not covered by this gate.
            pass


def check_inferred_shapes(ops: Sequence[OpBase], ctx: OpContext) -> None:
    """Shape-Gate: validate that output shapes are concrete after infer_shapes.

    Every tensor output must have either non-negative dimensions or an
    explicit ``dim_param`` for each negative dimension.
    Non-tensor outputs (e.g. sequences) are skipped.

    ``ctx.shape()`` raises ``ShapeInferenceError`` for missing tensor
    shapes — that error propagates uncaught as a genuine gate failure.
    ``UnsupportedOpError`` is caught and ignored because it signals a
    non-tensor output that this gate does not cover.
    """
    for op in ops:
        name = op.primary_output_name
        if name is None:
            continue
        try:
            shape = ctx.shape(name)
        except UnsupportedOpError:
            # Non-tensor output (e.g. sequence) — not covered by this gate.
            continue
        try:
            value = ctx.graph.find_value(name)
        except KeyError:
            value = None
        for dim_index, dim in enumerate(shape):
            dim_param = None
            if (
                value is not None
                and isinstance(value.type, TensorType)
                and dim_index < len(value.type.dim_params)
            ):
                dim_param = value.type.dim_params[dim_index]
            if dim < 0 and not dim_param:
                raise ShapeInferenceError(
                    f"Op {op.kind} output '{name}' has negative dimension "
                    f"{dim} at index {dim_index} after shape inference"
                )
