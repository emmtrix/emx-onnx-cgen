# ImageDecoder operator support

## Problem

The `ImageDecoder` operator (opset 20) decodes an encoded image byte stream
(JPEG, PNG, BMP, TIFF, WebP, JPEG 2000, PNM) into an `uint8[H, W, C]` pixel
tensor. An earlier implementation rewrote the node to `Identity`/`Gather` at
import time and decoded the image with Pillow inside the verification
harness, so the generated C code never decoded anything and its input
interface silently changed from encoded bytes to decoded pixels.

That violated two invariants: the generated code did not implement the
operator it claimed to support, and verification exercised Pillow rather than
the generated code.

## Options considered

1. **Drop ImageDecoder support.** Honest, but loses coverage.
2. **Implement all decoders from scratch in generated C.** Bit-exact JPEG /
   JPEG 2000 / WebP decoding is far beyond the scope of this project.
3. **Delegate to established decoder libraries, selected per format at
   compile time.** The ONNX reference outputs are produced with Pillow, which
   itself wraps libjpeg-turbo, libwebp, libtiff and OpenJPEG — using the same
   libraries makes bit-exact verification achievable.

## Decision

Option 3. The operator is lowered to an `ImageDecoderOp` whose kernel sniffs
the image format at runtime via magic bytes (the format is a runtime property
of the input bytes, not a compile-time property of the model) and dispatches
to a decoder implementation:

- `--image-decoder-libs` is a comma-separated priority list (default `stb`);
  for each format the first listed library that supports it is compiled in.
- `stb` is a vendored, header-only, public-domain decoder
  (`src/emx_onnx_cgen/third_party/stb_image.h`, emitted next to the generated
  C file) covering BMP/JPEG/PNG/PNM without system dependencies. Its JPEG
  decoder is *not* bit-exact with libjpeg, so JPEG test models select
  `libjpeg-turbo` instead.
- `libjpeg-turbo`, `libwebp`, `libtiff` and `openjpeg` are system libraries;
  the registry in `src/emx_onnx_cgen/codegen/image_decoder_libs.py` maps each
  to its formats, linker flags and (for OpenJPEG) a pkg-config module for
  include paths. `verify` applies the flags automatically; `compile` reports
  them.
- All decoders produce row-major RGB8; `pixel_format` conversion (BGR swap,
  ITU-R BT.601 grayscale matching Pillow's `convert("L")`) happens in the
  node kernel. Undecodable inputs (unknown format, no configured decoder,
  dimension mismatch with the declared static output shape) zero-fill the
  output deterministically.

## Consequences

- All nine `test_image_decoder_*` corpus models verify bit-exactly
  (`max abs diff 0`) against the ONNX-provided reference outputs, now
  exercising the generated C decoder instead of Pillow.
- The output shape must be static in the model (`H`, `W`, `C` declared); this
  matches the compiler-wide static-shape requirement.
- CI installs `libjpeg-turbo8-dev`, `libopenjp2-7-dev`, `libtiff-dev` and
  `libwebp-dev`; the per-model library selection is recorded in the
  expectation JSON via `extra_cli_args`.
- New decoder libraries can be added by extending the registry and the
  support-code template without touching lowering or CLI logic.
