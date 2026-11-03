# Per-node type inference during lowering

## Problem

Lowering used to run as separate whole-graph phases:

```
lower all nodes → validate all → infer_types all → infer_shapes all
```

Lowering handlers read input dtypes and shapes from the `GraphContext`, but
the IR's own `infer_types` / `infer_shapes` hooks only ran after every node had
been lowered. Any value ONNX shape inference left untyped therefore could not
be consumed by a later node, even when the producing IR op knows its output
type. This mostly affects contrib ops (e.g. `com.microsoft`) that have no ONNX
schema and whose outputs feed other nodes instead of graph outputs.

## Options

1. Keep the phases and work around missing types per op, e.g. by rewriting
   contrib nodes into the default domain before ONNX shape inference, or by
   peeking at producer nodes inside lowering handlers.
2. Register contrib op schemas with ONNX so its shape inference covers them.
   This duplicates IR knowledge in another place and depends on ONNX's schema
   registration API.
3. Run `validate` → `infer_types` → `infer_shapes` for each op right after it
   is lowered, in topological order.

## Decision

Option 3. `Compiler._lower_nodes` now lowers each node and immediately runs the
op's validation and inference hooks. Their results are written to the shared
`GraphContext` caches, so later lowering handlers see IR-inferred types. The
type and shape gates (`check_inferred_types`, `check_inferred_shapes`) still
run over the whole graph afterwards.

## Consequences

- An op's `infer_types` / `infer_shapes` is now the single place that can make
  an output type available to downstream lowering; ops without these hooks
  still rely on ONNX shape inference.
- Errors are reported in node order: an `infer_types` failure on an early node
  is raised before a lowering failure on a later node.
- `validate` of a node may now observe IR-inferred types of earlier nodes
  instead of only ONNX-declared ones.
