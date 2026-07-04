# Operator-Specific Notes

Some operators emit code with structure that is more specialized than the
general loop-nest patterns described in
[`docs/output-format.md`](output-format.md). This document collects noteworthy
algorithmic choices, implementation details, and limitations per operator
(or operator family).

The generated [`SUPPORT_OPS.md`](../SUPPORT_OPS.md) table links each operator
to its section in this document. The per-operator links and CLI options shown
there are maintained in [`docs/operator-metadata.json`](operator-metadata.json).

## FFT-Related Operators (DFT, STFT)

`DFT` and `STFT` are emitted as explicit FFT-based kernels over real/imaginary
lanes. They automatically choose between a
[Stockham FFT](https://en.wikipedia.org/wiki/Stockham_FFT) with radix-2 and
radix-4 stages and a direct DFT; `STFT` adds explicit framing and optional
windowing around the same spectral kernel. A concrete example is
[`tests/golden/op_dft_dft_stockham.c`](../tests/golden/op_dft_dft_stockham.c).

## Convolution Family (Conv, ConvTranspose, DeformConv)

**Conv** is implemented as a direct loop-based convolution (nested loops over
batch, group, output channel, spatial dimensions, input channel, and kernel
elements). It does *not* use im2col + GEMM. Padding and dilation are computed
inline for each output element. Grouped convolution divides the output channels
by the group count.

**ConvTranspose** uses a scatter-accumulate approach: the output is initialized
with bias, then for each input spatial position and kernel element, the
contribution is scattered to the corresponding output location
(`out_idx = in_idx * stride + kernel_idx * dilation - pad_begin`). Bounds
checking ensures only valid output positions are written.

**DeformConv** extends convolution with learned spatial offsets. For each kernel
position, a 2D offset is read from the offset tensor to perturb the sampling
location. The value at the offset position is computed via
[bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation)
over the four surrounding integer coordinates. An optional multiplicative mask
gates each deformed sample.

## Recurrent Networks (LSTM, GRU, RNN)

**LSTM** implements the standard
[LSTM cell](https://en.wikipedia.org/wiki/Long_short-term_memory) with input,
forget, and output gates. The internal weight layout follows gate order
`[i, o, f, c]` for both input and recurrence weights. Peephole connections are
supported optionally: the input and forget gates receive the previous cell
state, while the output gate receives the updated cell state. An optional
input-forget mode forces `f = 1 − i`, coupling the two gates.

**GRU** implements the
[Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) with
reset and update gates. The `linear_before_reset` attribute selects between two
variants: when false, the reset gate is applied element-wise before the
recurrent transform (`R_h @ (r * H_prev)`); when true, the linear transform is
computed first and the reset gate is applied afterwards (`r * (R_h @ H_prev)`).

**RNN** implements a simple single-activation recurrence:
`H_new = activation(W @ x + R @ H_prev + bias)`.

All three support configurable activation functions per gate and bidirectional
processing. A concrete RNN example is
[`tests/golden/op_rnn_rnn.c`](../tests/golden/op_rnn_rnn.c).

## Attention and Transformers

**Attention** implements multi-head attention with support for
[grouped-query attention (GQA)](https://arxiv.org/abs/2305.13245), where the
number of query heads can exceed the number of key/value heads (each KV head is
shared across a group of query heads). The computation follows:
QK matmul → optional scaling/softcap → mask application → softmax → AV matmul.
Softmax uses the max-subtraction trick for numerical stability (see
[Softmax](#softmax-and-logsoftmax) below). KV-cache inputs (`past_key`,
`past_value`) are supported for incremental autoregressive decoding, and causal
masking can suppress upper-triangular attention scores.

**RotaryEmbedding** applies
[Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864) by
rotating consecutive pairs of dimensions in the head representation using
precomputed cos/sin tables. Only the first `rotary_dim` elements are rotated;
the remainder is passed through unchanged. Interleaved and non-interleaved
layouts for element pairs are both supported.

## Spatial Sampling (Resize, GridSample, RoiAlign)

**Resize** supports seven coordinate transformation modes (`half_pixel`,
`align_corners`, `asymmetric`, `pytorch_half_pixel`, `tf_crop_and_resize`,
`tf_half_pixel_for_nn`, `half_pixel_symmetric`) and three interpolation modes
(`nearest` with four rounding sub-modes, `linear`/bilinear, `cubic` with a
configurable spline coefficient, default −0.75). An optional antialias mode
applies weighted coefficient normalization for downsampling.

**GridSample** performs learnable spatial sampling from a normalized coordinate
grid (values in [−1, 1]). It supports nearest, bilinear, and bicubic
interpolation, with three padding modes: `zeros` (output 0 outside bounds),
`border` (clamp to boundary), and `reflection` (reflect at boundaries).

**RoiAlign** maps each region-of-interest to a fixed-size output via adaptive
sub-grid sampling. Within each output bin, multiple points are sampled using
bilinear interpolation and combined via average or max pooling. The sampling
ratio is configurable or auto-computed from the ROI size.

## Normalization Variants

All normalization operators share the pattern `(x − mean) / sqrt(variance + ε)`,
but differ in which dimensions statistics are computed over:

- **BatchNormalization**: per-channel statistics across batch and spatial
  dimensions; uses running mean/variance with momentum during training.
- **LayerNormalization**: per-sample statistics across a configurable axis and
  all subsequent dimensions; no running statistics. The implementation supports
  optional [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
  for improved floating-point precision.
- **GroupNormalization**: channels are split into groups; statistics are computed
  per sample per group across all spatial dimensions.
- **InstanceNormalization**: equivalent to GroupNormalization with `num_groups =
  num_channels` (each channel is its own group).
- **RMSNormalization**: uses root-mean-square instead of variance
  (`RMS = sqrt(mean(x²) + ε)`), omitting mean centering. Popular in LLM
  architectures.

## Softmax and LogSoftmax

Both operators use the
[max-subtraction trick](https://en.wikipedia.org/wiki/LogSumExp) for numerical
stability: `softmax(x)_i = exp(x_i − max(x)) / Σ exp(x_j − max(x))`.
Subtracting the maximum value before exponentiation prevents floating-point
overflow.

## Selection and Filtering (TopK, NonMaxSuppression)

**TopK** uses an insertion-sort algorithm over a fixed-size window of K
elements. After filling the initial K slots, each remaining element is
streamed through the sorted window and inserted in order if it qualifies.
Both values and indices are emitted.

**NonMaxSuppression** implements standard greedy NMS: candidates are sorted by
score using insertion sort, then iteratively selected while suppressing
overlapping boxes above the
[IoU](https://en.wikipedia.org/wiki/Jaccard_index) threshold. Both corner
format (`[x_min, y_min, x_max, y_max]`) and center-point format
(`[cx, cy, w, h]`) are supported. Parameters include `max_output_boxes_per_class`,
`iou_threshold`, and `score_threshold`.

## Dynamic-Output Operators (Unique, NonZero, Compress)

These operators produce outputs whose size depends on input data, not just
input shape:

- **Unique**: uses linear-search deduplication (not hash-based). For each input
  element, existing unique values are scanned for a match. Optional bubble sort
  is applied when `sorted=1`. Outputs include unique values, indices, inverse
  indices, and counts.
- **NonZero**: iterates over all input elements and stores multi-dimensional
  indices for non-zero entries. Output shape is `[rank, nnz]`.
- **Compress**: applies a boolean mask to select elements along an axis (or
  flat). The output size equals the number of true mask entries.

## Reconstruction (Col2Im, MaxUnpool)

**Col2Im** reverses the im2col transformation by scattering column values back
into an image tensor. The output is initialized to zero, and overlapping
positions are accumulated (summed). Stride, dilation, and padding are respected.

**MaxUnpool** reconstructs a larger tensor from MaxPool outputs using the stored
pooling indices. For each input value, the corresponding flat index from the
indices tensor determines the output position. The remaining output positions
are initialized to zero. 1D, 2D, and 3D spatial ranks are handled separately.

## Tree-Based Models (TreeEnsemble, TreeEnsembleClassifier)

Trees are traversed via explicit if-else chains (not table-driven dispatch).
Each node stores a feature index, comparison mode, and split threshold. Six
comparison modes are supported (≤, <, ≥, >, =, ≠) plus set-membership branching.
The traversal follows true/false branches from tree roots until a leaf is
reached. Leaf contributions are summed across all trees in the ensemble.

## Random Number Generation (RandomUniform, Bernoulli)

**RandomUniform** uses the
[xorshift64*](https://en.wikipedia.org/wiki/Xorshift#xorshift*) PRNG algorithm
with the shift triple (12, 25, 27) and multiplier `0x2545F4914F6CDD1D`. The
64-bit state is mapped to [0, 1) and then scaled to the requested range
`[low, high)`. The seed defaults to `0x243F6A8885A308D3` (derived from the
fractional digits of π) if not provided.

**Bernoulli** samples from a Bernoulli distribution using the same PRNG
infrastructure with a per-element probability threshold.

## Einsum (Limited Patterns)

Einsum is *not* implemented as a general-purpose tensor contraction engine.
Instead, only a fixed set of patterns is recognized and lowered to specialized
loop nests:

| Pattern | Semantics |
| --- | --- |
| `->` | Scalar reduction of a single tensor |
| `ij->i` | Row-wise sum (contract j) |
| `ij->ji` | Matrix transpose |
| `i,i->` | Vector dot product |
| `bij,bjk->bik` | Batched matrix multiplication |
| `...ii->...i` | Batched diagonal extraction |

Unrecognized patterns raise an error at lowering time.

## Mel-Frequency Filter Banks (MelWeightMatrix)

Generates a triangular mel-scale filter bank. Frequency-to-mel conversion
uses the [Slaney formula](https://en.wikipedia.org/wiki/Mel_scale):
`mel(f) = 2595 × log₁₀(1 + f / 700)`. For each mel bin, a triangular window
with linear rise and fall slopes is constructed over the corresponding
DFT frequency bins.

## Quantization (QuantizeLinear, DequantizeLinear)

**QuantizeLinear** quantizes floating-point values using
`round(value / scale) + zero_point`, where rounding uses
[round-half-to-even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)
(banker's rounding via `nearbyint`). The result is clamped to the target
integer type's range.

**DequantizeLinear** reconstructs floating-point values via
`(quantized − zero_point) × scale`.

## Text Processing (TfIdfVectorizer, LabelEncoder)

**TfIdfVectorizer** implements n-gram matching by scanning input token
sequences for matches against a flattened vocabulary pool. Configurable
parameters include `min_gram_length`, `max_gram_length`, and `max_skip_count`.
Weighting modes include raw counts, binary indicators, and TF-IDF weights.

**LabelEncoder** uses a linear-scan lookup table (not hash-based): input
values are compared against each key entry sequentially, and the corresponding
value (or a default) is returned.

## Image Decoding (ImageDecoder)

`ImageDecoder` decodes an encoded image byte stream (JPEG, PNG, BMP, TIFF,
WebP, JPEG 2000, PNM) into an `uint8[H, W, C]` pixel tensor. Because the
image format is a runtime property of the input bytes — not a compile-time
property of the model — the generated kernel detects the format at runtime
via magic bytes and dispatches to a decoder library compiled into `model.c`.

Which library backs each format is selected with `--image-decoder-libs`, a
comma-separated priority list (first library supporting a format wins;
default: `stb`):

| Library | Formats | Build requirements |
| --- | --- | --- |
| `stb` | bmp, jpeg, png, pnm | none — bundled `stb_image.h` is emitted next to the generated C file |
| `libjpeg-turbo` | jpeg | `libjpeg-turbo8-dev`, links `-ljpeg` |
| `libwebp` | webp | `libwebp-dev`, links `-lwebp` |
| `libtiff` | tiff | `libtiff-dev`, links `-ltiff` |
| `openjpeg` | jpeg2000 | `libopenjp2-7-dev`, links `-lopenjp2` (include path via pkg-config) |

`verify` applies the required compiler/linker flags automatically; `compile`
emits the support header and reports the flags. `stb` needs no system
libraries but its JPEG decoder is not bit-exact with libjpeg — the ONNX
reference outputs are produced with Pillow (which wraps libjpeg-turbo,
libwebp, libtiff, and OpenJPEG), so bit-exact verification of
JPEG/WebP/TIFF/JPEG 2000 models requires selecting the matching library
(e.g. `--image-decoder-libs libjpeg-turbo,stb`).

All decoders produce row-major RGB8; `pixel_format` conversion happens in the
node kernel: `BGR` swaps channels, `Grayscale` applies the ITU-R BT.601 luma
formula `(19595·R + 38470·G + 7471·B + 32768) >> 16`, matching Pillow's
`convert("L")` bit-exactly. The decoded output shape must be declared
statically in the model (`H`, `W`, `C`). Undecodable inputs — unknown format,
no decoder configured for the detected format, or decoded dimensions that do
not match the declared output shape — zero-fill the output deterministically.

## Control Flow (Loop, Scan)

Loop and Scan operators are lowered via pattern matching on the subgraph
body, not via general-purpose subgraph inlining. Recognized patterns
(e.g. `LoopRange` for simple counter loops) are mapped directly to C loop
constructs. Unrecognized body structures raise an error at lowering time.
