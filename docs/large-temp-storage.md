# Storage of large temporaries

## Problem

Temporaries above `--large-temp-threshold` (default 1024 bytes) must leave the
stack, otherwise large models overflow small embedded stacks. Up to #638 they
were all `static`. #638 added heap allocation for temporaries whose size
depends on a runtime dimension, but its check ran before the `static` branch
and ignored whether the size is fixed, so every large temporary went to the
heap (`malloc`/`free` on every call). #785 removed the then-dead `static`
branch and documented the heap behavior.

## Options

- **Heap for all large temporaries** (#638 to now). Reentrant, memory only
  taken during a call, but every static-shape model needs an allocator and
  pays `malloc`/`free` on every call; a failed allocation silently returns.
- **Static for fixed-size, heap for runtime-sized** (before #638, intended by
  #638). No allocator for static-shape models, the size shows up in the
  linker map; the model is not reentrant.
- **Stack for everything** (`--large-temp-threshold 0` or
  `EMX_LARGE_TEMPS_ON_STACK`). No allocator and no static storage, but the
  stack has to hold all temporaries.

## Decision

Fixed-size large temporaries are `static` by default; only runtime-sized ones
use `malloc`/`free` (`CEmitter._temp_buffer_is_static`,
`CEmitter._temp_buffer_uses_heap`). `--large-temp-storage malloc` selects the
heap for the fixed-size ones too. The stack variant stays behind
`EMX_LARGE_TEMPS_ON_STACK`.

The primary targets are embedded systems where a model runs in one task and an
allocator is unwanted. Sequence temporaries and `Tokenizer` buffers already
have static storage (and the state arrays of `--state-pair`, #811), so the
generated code was not reentrant anyway.

## Consequences

- The generated code is not reentrant (`docs/output-format.md`, "Reentrancy").
- Static temporaries stay reserved for the program's lifetime, one set per
  model, instead of being taken only during a call; for large models this
  moves a lot of memory into `.bss`. Applications that need the memory back
  between calls use `--large-temp-storage malloc`, or compile with
  `EMX_LARGE_TEMPS_ON_STACK` and size the stack accordingly.
- With `malloc`, all heap temporaries are allocated at the start of the call
  and freed at its end, so the peak is the sum of all of them, not the largest
  set live at the same time.
- Static-shape models no longer include `<stdlib.h>` for temporaries.
