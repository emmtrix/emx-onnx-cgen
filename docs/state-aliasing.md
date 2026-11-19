# State aliasing: one persistent buffer per input/output pair

## Problem

Step-wise models such as a decoder step with a KV cache take the state as a
graph input (`past_k_i`) and return the updated state as a graph output
(`new_k_i`). Each call copies the whole cache in, copies it into the output and
the caller copies it back for the next call. For the Tiny char-level decoder
(`[4,128,24]` floats per K/V and layer) that is several 100 KiB of memory
traffic per token, and the cache needs two buffers. On the AURIX TC4x PPU the
small vector memory cannot hold either copy.

ONNX has no standard for state between calls:

- [In-place KV Cache for Attention](https://onnx.ai/onnx/technical/kv_cache.html)
  leaves input/output aliasing to the backend. Its pattern is a fixed-size
  cache written with `TensorScatter` (opset 24) and read by `Attention` with
  `nonpad_kv_seqlen`. Its condition: nothing may read the `TensorScatter`
  input after the write.
- onnx/onnx#8370 is an open umbrella issue with a `StateBinding` sketch
  (`input_value_name` / `output_value_name` / initialization).
- OpenVINO `MakeStateful` takes an explicit Parameter→Result name map and
  requires the same shape and type. ORT GenAI `past_present_share_buffer` pairs
  past/present through configured name lists.

## Decision

### Explicit pairs

The user names the pairs: `--state-pair past_k_0=new_k_0` (repeatable or
comma-separated), accepted by `compile` and `verify` alike (compile/verify
parity), or the model metadata key `emx_onnx_cgen.state_pairs` with the same
syntax. The compiler reads the metadata itself (`CompilerOptions.state_pairs`
is `None` by default), so every entry point yields the same ABI for a model;
`--state-pair` replaces the metadata pairs and `--no-state-pairs`
(`state_pairs=()`) ignores them. The ONNX backend always passes `()` because it
runs single calls with explicit inputs and outputs. The compiler never infers
pairs from name patterns: a wrong guess would silently change the ABI.

### Pass: `apply_state_aliasing` (`src/emx_onnx_cgen/state_aliasing.py`)

The pass runs on the lowered model (after dead-op pruning, so it sees the final
op order) and is a pure function `LoweredModel -> LoweredModel`:

1. **Type check.** Input and output must be graph tensors (no sequences, not
   optional) with the same static shape and dtype.
2. **Legality check.** Let `W` be the op that writes the output. With one
   buffer, `W` overwrites the input. So:
   - no op after `W` may read the input;
   - `W` itself may read the input only through input fields whose kernel
     slot is in-place safe (`OpBase.in_place_input_fields`); a second slot
     reading the same buffer (e.g. `TensorScatter(past, past, idx)`) is
     rejected. Today the safe slots are every same-index
     elementwise op (input shape equal to the output shape, no broadcast) and
     `TensorScatter` (`past_cache -> output`).

   Ops before `W` may read the input freely, so the current Tiny export
   `past*(1-onehot) + fresh*onehot` is legal, too.
3. **Rename.** Every op reference to the output is renamed to the input. The
   emitter then sees one value per state buffer; `LoweredModel.state_bindings`
   records the pairs.

Any failed check raises `StateAliasingError` with the pair, node index, op type,
node name, the offending names and shapes, and a hint. There is no silent
fallback to a copy.

### Storage: a caller-owned state struct

```c
typedef struct {
    float past_k_0[4][128][24]; /* past_k_0 -> new_k_0 */
    ...
} model_state;

void model_state_reset(model_state *state);            /* zero-fills */
void model(model_state *restrict state, /* other inputs */, /* other outputs */);
```

The entry point loses the paired inputs/outputs and takes `model_state *`
instead. Kernels receive `state->past_k_0` like any other array.

Alternatives considered:

- **One array parameter per pair.** As flexible for placement, but the
  signature grows with the layer count (2 parameters per layer for Tiny) and
  `reset` needs all of them again.
- **A generated global/static with a section macro.** Placement would need a
  compiler-specific hook (`__attribute__((section))` vs. TASKING
  `#pragma section`), and the model would become non-reentrant (only one
  sequence at a time).

The struct keeps the ABI to one handle, lets the caller decide placement with
the toolchain's own mechanism (section attribute, pragma, linker script, or a
heap buffer), and allows several independent sequences (one struct each). It
requires static state shapes, which a fixed-size cache has anyway.

`model_state_reset()` zero-fills every member; the caller can also write
initial values directly into the struct.

### TensorScatter in place

When `past_cache` and `output` are the same buffer (`TensorScatterOp.in_place`),
the kernel drops the `past_cache` parameter and the full-cache copy loop; it
only writes the updated rows (`tests/golden/kv_cache_state_model.c`).

### Verification

The reference run keeps the original model with explicit inputs and outputs.
`verify --state-pair ... --state-steps N` (default 3):

- generates one initial state per pair and N sets of the other inputs.
  Integer inputs that an op uses as an unchecked index declare their valid
  range (`OpBase.index_input_bounds`, e.g. TensorScatter `write_indices` in
  `[0, S - L]` for linear mode) and are drawn from it; other integers come
  from `[0, 4)`. The C testbench uses the same bounds when it fills inputs
  randomly (`compile --emit-testbench` without data);
- runs the reference N times, feeding each step's state outputs back as state
  inputs;
- the generated testbench calls `model_state_reset()`, loads the initial state
  into its state struct and calls the model N times through that buffer;
- compares every output of every step (a leading step axis on both sides).

The compiler receives these values as `CompilerOptions.state_testbench`
(`StateTestbenchData`: `steps`, `initial_state`, `step_inputs` with a leading
step axis) instead of `testbench_inputs`. The testbench picks its own
identifiers (step counter, state variable, collected outputs, loop indices)
so that they never clash with the model's value names.

`--test-data-dir` and `--per-node-accuracy` are rejected together with state
pairs (use `--no-state-pairs` for a model whose metadata declares pairs).

## Consequences

- The KV cache exists once, in memory the caller chooses, and each step only
  writes the new rows.
- Models whose attention reads `past` instead of the `TensorScatter` output are
  rejected with a hint; they need re-export (read the updated cache), or no
  pair.
- A subgraph (`If`/`Loop` body) that captures the state input counts as a read
  by its node.
- Limitations: static shapes only; tensor (non-string) state only; the
  multi-step testbench supports neither dynamic dimensions nor sequence or
  optional I/O.
