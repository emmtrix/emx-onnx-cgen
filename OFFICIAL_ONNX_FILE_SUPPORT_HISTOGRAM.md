# Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Out of tolerance (max ULP 4294967295) | 21 | ██████████████████████████████ |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 18 | ██████████████████████████ |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | ██████████████████████████ |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 18 | ██████████████████████████ |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | ██████████████████████████ |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 15 | █████████████████████ |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 15 | █████████████████████ |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 15 | █████████████████████ |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 15 | █████████████████████ |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 12 | █████████████████ |
| Where output shape must be (1, 1), got (1,) | 10 | ██████████████ |
| Testbench execution failed: exit code -11 (signal 11: SIGSEGV) | 9 | █████████████ |
| AveragePool has unsupported attributes | 6 | █████████ |
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 6 | █████████ |
| Unsupported op CenterCropPad | 6 | █████████ |
| And expects identical input/output shapes | 5 | ███████ |
| Unsupported op Col2Im | 5 | ███████ |
| Unsupported op AffineGrid | 4 | ██████ |
| Unsupported op If | 4 | ██████ |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 4 | ██████ |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | ██████ |
| Unsupported op Compress | 4 | ██████ |
| Out of tolerance (max ULP 1818802) | 3 | ████ |
| AveragePool supports auto_pad=NOTSET only | 3 | ████ |
| Unsupported op Bernoulli | 3 | ████ |
| Unsupported op RandomUniformLike | 3 | ████ |
| Unsupported op Adagrad | 2 | ███ |
| Unsupported op Adam | 2 | ███ |
| Unsupported op TreeEnsemble | 2 | ███ |
| AveragePool expects 2D kernel_shape | 2 | ███ |
| AveragePool supports ceil_mode=0 only | 2 | ███ |
| Unsupported op DeformConv | 2 | ███ |
| BatchNormalization must have 5 inputs and 1 output | 2 | ███ |
| BitwiseAnd expects identical input/output shapes | 2 | ███ |
| Unsupported op BitwiseNot | 2 | ███ |
| BitwiseOr expects identical input/output shapes | 2 | ███ |
| BitwiseXor expects identical input/output shapes | 2 | ███ |
| Unsupported op BlackmanWindow | 2 | ███ |
| Cast input and output shapes must match | 2 | ███ |
| Out of tolerance (max ULP 1084227585) | 2 | ███ |
| Out of tolerance (max ULP 2143208269) | 1 | █ |
| Unsupported op ArrayFeatureExtractor | 1 | █ |
| Unsupported op Binarizer | 1 | █ |
| Failed to build testbench: /tmp/tmp9xuzzb95/model.c: In function ‘node18_identity’:
/tmp/tmp9xuzzb95/model.c:532:12: error: ‘i0’ undeclared (first use in this function)
  532 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmp9xuzzb95/model.c:532:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmp9xuzzb95/model.c: In function ‘node19_sqrt’:
/tmp/tmp9xuzzb95/model.c:544:12: error: ‘i0’ undeclared (first use in this function)
  544 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmp9xuzzb95/model.c: In function ‘node20_cast’:
/tmp/tmp9xuzzb95/model.c:557:12: error: ‘i0’ undeclared (first use in this function)
  557 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmp9xuzzb95/model.c: In function ‘model’:
/tmp/tmp9xuzzb95/model.c:1498:102: warning: passing argument 2 of ‘node47_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1498 |     node47_mul(tmp8_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QReshaped, tmp20_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_ScaleFactorF, tmp47_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QScaled);
      |                                                                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                      |
      |                                                                                                      float *
/tmp/tmp9xuzzb95/model.c:1100:84: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float *’
 1100 | static inline void node47_mul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][4][8], float output[restrict 2][3][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1499:104: warning: passing argument 2 of ‘node48_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1499 |     node48_mul(tmp46_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_KTranspose, tmp20_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_ScaleFactorF, tmp48_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_KScaled);
      |                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                        |
      |                                                                                                        float *
/tmp/tmp9xuzzb95/model.c:1120:84: note: expected ‘const float (* restrict)[3][8][6]’ but argument is of type ‘float *’
 1120 | static inline void node48_mul(const float input0[restrict 2][3][8][6], const float input1[restrict 2][3][8][6], float output[restrict 2][3][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1500:189: warning: passing argument 3 of ‘node49_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1500 |     node49_matmul(tmp47_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QScaled, tmp48_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_KScaled, tmp49_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                                                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                                             |
      |                                                                                                                                                                                             float (*)[6]
/tmp/tmp9xuzzb95/model.c:1140:122: note: expected ‘float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1140 | static inline void node49_matmul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][8][6], float output[restrict 2][3][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1501:17: warning: passing argument 1 of ‘node50_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1501 |     node50_cast(tmp49_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QKAttnWeight, tmp50_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmp9xuzzb95/model.c:1165:44: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1165 | static inline void node50_cast(const float input0[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1502:104: warning: passing argument 2 of ‘node51_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1502 |     node51_add(tmp50_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QKAttnCast, tmp27_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_AttnBiasT, tmp51_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                        |
      |                                                                                                        float (*)[6]
/tmp/tmp9xuzzb95/model.c:1185:84: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1185 | static inline void node51_add(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1507:200: warning: passing argument 3 of ‘node56_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1507 |     node56_matmul(tmp55_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_SoftmaxOut, tmp45_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_VAttentionInput, tmp56_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                                                        |
      |                                                                                                                                                                                                        float (*)[10]
/tmp/tmp9xuzzb95/model.c:1304:123: note: expected ‘float (* restrict)[3][4][10]’ but argument is of type ‘float (*)[10]’
 1304 | static inline void node56_matmul(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][6][10], float output[restrict 2][3][4][10]) {
      |                                                                                                                     ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp9xuzzb95/model.c:1508:22: warning: passing argument 1 of ‘node57_transpose’ from incompatible pointer type [-Wincompatible-pointer-types]
 1508 |     node57_transpose(tmp56_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_YPreReshape, tmp57_Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_YTranspose);
      |                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                      |
      |                      float (*)[10]
/tmp/tmp9xuzzb95/model.c:1329:49: note: expected ‘const float (* restrict)[3][4][10]’ but argument is of type ‘float (*)[10]’
 1329 | static inline void node57_transpose(const float input0[restrict 2][3][4][10], float output[restrict 2][4][3][10]) {
      |                                     ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Failed to build testbench: /tmp/tmpc933pfnn/model.c: In function ‘node18_identity’:
/tmp/tmpc933pfnn/model.c:532:12: error: ‘i0’ undeclared (first use in this function)
  532 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmpc933pfnn/model.c:532:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmpc933pfnn/model.c: In function ‘node19_sqrt’:
/tmp/tmpc933pfnn/model.c:544:12: error: ‘i0’ undeclared (first use in this function)
  544 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmpc933pfnn/model.c: In function ‘node20_cast’:
/tmp/tmpc933pfnn/model.c:557:12: error: ‘i0’ undeclared (first use in this function)
  557 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmpc933pfnn/model.c: In function ‘model’:
/tmp/tmpc933pfnn/model.c:1498:89: warning: passing argument 2 of ‘node47_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1498 |     node47_mul(tmp8_Attention_test_attention_3d_gqa_scaled_expanded_function_QReshaped, tmp20_Attention_test_attention_3d_gqa_scaled_expanded_function_ScaleFactorF, tmp47_Attention_test_attention_3d_gqa_scaled_expanded_function_QScaled);
      |                                                                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                         |
      |                                                                                         float *
/tmp/tmpc933pfnn/model.c:1100:84: note: expected ‘const float (* restrict)[9][4][8]’ but argument is of type ‘float *’
 1100 | static inline void node47_mul(const float input0[restrict 2][9][4][8], const float input1[restrict 2][9][4][8], float output[restrict 2][9][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1499:91: warning: passing argument 2 of ‘node48_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1499 |     node48_mul(tmp46_Attention_test_attention_3d_gqa_scaled_expanded_function_KTranspose, tmp20_Attention_test_attention_3d_gqa_scaled_expanded_function_ScaleFactorF, tmp48_Attention_test_attention_3d_gqa_scaled_expanded_function_KScaled);
      |                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                           |
      |                                                                                           float *
/tmp/tmpc933pfnn/model.c:1120:84: note: expected ‘const float (* restrict)[9][8][6]’ but argument is of type ‘float *’
 1120 | static inline void node48_mul(const float input0[restrict 2][9][8][6], const float input1[restrict 2][9][8][6], float output[restrict 2][9][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1500:163: warning: passing argument 3 of ‘node49_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1500 |     node49_matmul(tmp47_Attention_test_attention_3d_gqa_scaled_expanded_function_QScaled, tmp48_Attention_test_attention_3d_gqa_scaled_expanded_function_KScaled, tmp49_Attention_test_attention_3d_gqa_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                   |
      |                                                                                                                                                                   float (*)[6]
/tmp/tmpc933pfnn/model.c:1140:122: note: expected ‘float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
 1140 | static inline void node49_matmul(const float input0[restrict 2][9][4][8], const float input1[restrict 2][9][8][6], float output[restrict 2][9][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1501:17: warning: passing argument 1 of ‘node50_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1501 |     node50_cast(tmp49_Attention_test_attention_3d_gqa_scaled_expanded_function_QKAttnWeight, tmp50_Attention_test_attention_3d_gqa_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmpc933pfnn/model.c:1165:44: note: expected ‘const float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
 1165 | static inline void node50_cast(const float input0[restrict 2][9][4][6], float output[restrict 2][9][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1502:91: warning: passing argument 2 of ‘node51_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1502 |     node51_add(tmp50_Attention_test_attention_3d_gqa_scaled_expanded_function_QKAttnCast, tmp27_Attention_test_attention_3d_gqa_scaled_expanded_function_AttnBiasT, tmp51_Attention_test_attention_3d_gqa_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                           |
      |                                                                                           float (*)[6]
/tmp/tmpc933pfnn/model.c:1185:84: note: expected ‘const float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
 1185 | static inline void node51_add(const float input0[restrict 2][9][4][6], const float input1[restrict 2][9][4][6], float output[restrict 2][9][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1507:174: warning: passing argument 3 of ‘node56_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1507 |     node56_matmul(tmp55_Attention_test_attention_3d_gqa_scaled_expanded_function_SoftmaxOut, tmp45_Attention_test_attention_3d_gqa_scaled_expanded_function_VAttentionInput, tmp56_Attention_test_attention_3d_gqa_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                              |
      |                                                                                                                                                                              float (*)[8]
/tmp/tmpc933pfnn/model.c:1304:122: note: expected ‘float (* restrict)[9][4][8]’ but argument is of type ‘float (*)[8]’
 1304 | static inline void node56_matmul(const float input0[restrict 2][9][4][6], const float input1[restrict 2][9][6][8], float output[restrict 2][9][4][8]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpc933pfnn/model.c:1508:22: warning: passing argument 1 of ‘node57_transpose’ from incompatible pointer type [-Wincompatible-pointer-types]
 1508 |     node57_transpose(tmp56_Attention_test_attention_3d_gqa_scaled_expanded_function_YPreReshape, tmp57_Attention_test_attention_3d_gqa_scaled_expanded_function_YTranspose);
      |                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                      |
      |                      float (*)[8]
/tmp/tmpc933pfnn/model.c:1329:49: note: expected ‘const float (* restrict)[9][4][8]’ but argument is of type ‘float (*)[8]’
 1329 | static inline void node57_transpose(const float input0[restrict 2][9][4][8], float output[restrict 2][4][9][8]) {
      |                                     ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Failed to build testbench: /tmp/tmpk_gelunb/model.c: In function ‘node18_identity’:
/tmp/tmpk_gelunb/model.c:532:12: error: ‘i0’ undeclared (first use in this function)
  532 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmpk_gelunb/model.c:532:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmpk_gelunb/model.c: In function ‘node19_sqrt’:
/tmp/tmpk_gelunb/model.c:544:12: error: ‘i0’ undeclared (first use in this function)
  544 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmpk_gelunb/model.c: In function ‘node20_cast’:
/tmp/tmpk_gelunb/model.c:557:12: error: ‘i0’ undeclared (first use in this function)
  557 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmpk_gelunb/model.c: In function ‘model’:
/tmp/tmpk_gelunb/model.c:1498:85: warning: passing argument 2 of ‘node47_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1498 |     node47_mul(tmp8_Attention_test_attention_3d_scaled_expanded_function_QReshaped, tmp20_Attention_test_attention_3d_scaled_expanded_function_ScaleFactorF, tmp47_Attention_test_attention_3d_scaled_expanded_function_QScaled);
      |                                                                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                     |
      |                                                                                     float *
/tmp/tmpk_gelunb/model.c:1100:84: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float *’
 1100 | static inline void node47_mul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][4][8], float output[restrict 2][3][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1499:87: warning: passing argument 2 of ‘node48_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1499 |     node48_mul(tmp46_Attention_test_attention_3d_scaled_expanded_function_KTranspose, tmp20_Attention_test_attention_3d_scaled_expanded_function_ScaleFactorF, tmp48_Attention_test_attention_3d_scaled_expanded_function_KScaled);
      |                                                                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                       |
      |                                                                                       float *
/tmp/tmpk_gelunb/model.c:1120:84: note: expected ‘const float (* restrict)[3][8][6]’ but argument is of type ‘float *’
 1120 | static inline void node48_mul(const float input0[restrict 2][3][8][6], const float input1[restrict 2][3][8][6], float output[restrict 2][3][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1500:155: warning: passing argument 3 of ‘node49_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1500 |     node49_matmul(tmp47_Attention_test_attention_3d_scaled_expanded_function_QScaled, tmp48_Attention_test_attention_3d_scaled_expanded_function_KScaled, tmp49_Attention_test_attention_3d_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                           |
      |                                                                                                                                                           float (*)[6]
/tmp/tmpk_gelunb/model.c:1140:122: note: expected ‘float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1140 | static inline void node49_matmul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][8][6], float output[restrict 2][3][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1501:17: warning: passing argument 1 of ‘node50_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1501 |     node50_cast(tmp49_Attention_test_attention_3d_scaled_expanded_function_QKAttnWeight, tmp50_Attention_test_attention_3d_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmpk_gelunb/model.c:1165:44: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1165 | static inline void node50_cast(const float input0[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1502:87: warning: passing argument 2 of ‘node51_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1502 |     node51_add(tmp50_Attention_test_attention_3d_scaled_expanded_function_QKAttnCast, tmp27_Attention_test_attention_3d_scaled_expanded_function_AttnBiasT, tmp51_Attention_test_attention_3d_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                       |
      |                                                                                       float (*)[6]
/tmp/tmpk_gelunb/model.c:1185:84: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1185 | static inline void node51_add(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1507:166: warning: passing argument 3 of ‘node56_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1507 |     node56_matmul(tmp55_Attention_test_attention_3d_scaled_expanded_function_SoftmaxOut, tmp45_Attention_test_attention_3d_scaled_expanded_function_VAttentionInput, tmp56_Attention_test_attention_3d_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                      |
      |                                                                                                                                                                      float (*)[8]
/tmp/tmpk_gelunb/model.c:1304:122: note: expected ‘float (* restrict)[3][4][8]’ but argument is of type ‘float (*)[8]’
 1304 | static inline void node56_matmul(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][6][8], float output[restrict 2][3][4][8]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpk_gelunb/model.c:1508:22: warning: passing argument 1 of ‘node57_transpose’ from incompatible pointer type [-Wincompatible-pointer-types]
 1508 |     node57_transpose(tmp56_Attention_test_attention_3d_scaled_expanded_function_YPreReshape, tmp57_Attention_test_attention_3d_scaled_expanded_function_YTranspose);
      |                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                      |
      |                      float (*)[8]
/tmp/tmpk_gelunb/model.c:1329:49: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float (*)[8]’
 1329 | static inline void node57_transpose(const float input0[restrict 2][3][4][8], float output[restrict 2][4][3][8]) {
      |                                     ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Out of tolerance (max ULP 2986210) | 1 | █ |
| Out of tolerance (max ULP 2044718) | 1 | █ |
| Out of tolerance (max ULP 4617623) | 1 | █ |
| ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 12, max supported IR version: 11
 | 1 | █ |
| Pad value input must be a scalar | 1 | █ |
| Out of tolerance (max ULP 3668711) | 1 | █ |
| Out of tolerance (max ULP 2004067) | 1 | █ |
| Failed to build testbench: /tmp/tmpwlbnefxy/model.c: In function ‘node13_identity’:
/tmp/tmpwlbnefxy/model.c:384:12: error: ‘i0’ undeclared (first use in this function)
  384 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmpwlbnefxy/model.c:384:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmpwlbnefxy/model.c: In function ‘node14_sqrt’:
/tmp/tmpwlbnefxy/model.c:396:12: error: ‘i0’ undeclared (first use in this function)
  396 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmpwlbnefxy/model.c: In function ‘node15_cast’:
/tmp/tmpwlbnefxy/model.c:409:12: error: ‘i0’ undeclared (first use in this function)
  409 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmpwlbnefxy/model.c: In function ‘model’:
/tmp/tmpwlbnefxy/model.c:1287:102: warning: passing argument 2 of ‘node42_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1287 |     node42_mul(tmp3_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QReshaped, tmp15_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_ScaleFactorF, tmp42_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QScaled);
      |                                                                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                      |
      |                                                                                                      float *
/tmp/tmpwlbnefxy/model.c:952:84: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float *’
  952 | static inline void node42_mul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][4][8], float output[restrict 2][3][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1288:104: warning: passing argument 2 of ‘node43_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1288 |     node43_mul(tmp41_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_KTranspose, tmp15_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_ScaleFactorF, tmp43_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_KScaled);
      |                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                        |
      |                                                                                                        float *
/tmp/tmpwlbnefxy/model.c:972:84: note: expected ‘const float (* restrict)[3][8][6]’ but argument is of type ‘float *’
  972 | static inline void node43_mul(const float input0[restrict 2][3][8][6], const float input1[restrict 2][3][8][6], float output[restrict 2][3][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1289:189: warning: passing argument 3 of ‘node44_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1289 |     node44_matmul(tmp42_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QScaled, tmp43_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_KScaled, tmp44_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                                                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                                             |
      |                                                                                                                                                                                             float (*)[6]
/tmp/tmpwlbnefxy/model.c:992:122: note: expected ‘float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
  992 | static inline void node44_matmul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][8][6], float output[restrict 2][3][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1290:17: warning: passing argument 1 of ‘node45_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1290 |     node45_cast(tmp44_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QKAttnWeight, tmp45_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmpwlbnefxy/model.c:1017:44: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1017 | static inline void node45_cast(const float input0[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1291:104: warning: passing argument 2 of ‘node46_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1291 |     node46_add(tmp45_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QKAttnCast, tmp22_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_AttnBiasT, tmp46_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                        |
      |                                                                                                        float (*)[6]
/tmp/tmpwlbnefxy/model.c:1037:84: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1037 | static inline void node46_add(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1296:200: warning: passing argument 3 of ‘node51_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1296 |     node51_matmul(tmp50_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_SoftmaxOut, tmp40_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_VAttentionInput, tmp51_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                                                        |
      |                                                                                                                                                                                                        float (*)[10]
/tmp/tmpwlbnefxy/model.c:1156:123: note: expected ‘float (* restrict)[3][4][10]’ but argument is of type ‘float (*)[10]’
 1156 | static inline void node51_matmul(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][6][10], float output[restrict 2][3][4][10]) {
      |                                                                                                                     ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmpwlbnefxy/model.c:1297:21: warning: passing argument 1 of ‘node52_identity’ from incompatible pointer type [-Wincompatible-pointer-types]
 1297 |     node52_identity(tmp51_Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_YPreReshape, Y);
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                     |
      |                     float (*)[10]
/tmp/tmpwlbnefxy/model.c:1180:48: note: expected ‘const float (* restrict)[3][4][10]’ but argument is of type ‘float (*)[10]’
 1180 | static inline void node52_identity(const float input0[restrict 2][3][4][10], float output[restrict 2][3][4][10]) {
      |                                    ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Out of tolerance (max ULP 2460850) | 1 | █ |
| Out of tolerance (max ULP 2363933) | 1 | █ |
| Out of tolerance (max ULP 1556056) | 1 | █ |
| Out of tolerance (max ULP 222) | 1 | █ |
| Out of tolerance (max ULP 6539519) | 1 | █ |
| Out of tolerance (max ULP 3777733) | 1 | █ |
| Failed to build testbench: /tmp/tmppzik46xq/model.c: In function ‘node13_identity’:
/tmp/tmppzik46xq/model.c:384:12: error: ‘i0’ undeclared (first use in this function)
  384 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmppzik46xq/model.c:384:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmppzik46xq/model.c: In function ‘node14_sqrt’:
/tmp/tmppzik46xq/model.c:396:12: error: ‘i0’ undeclared (first use in this function)
  396 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmppzik46xq/model.c: In function ‘node15_cast’:
/tmp/tmppzik46xq/model.c:409:12: error: ‘i0’ undeclared (first use in this function)
  409 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmppzik46xq/model.c: In function ‘model’:
/tmp/tmppzik46xq/model.c:1287:89: warning: passing argument 2 of ‘node42_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1287 |     node42_mul(tmp3_Attention_test_attention_4d_gqa_scaled_expanded_function_QReshaped, tmp15_Attention_test_attention_4d_gqa_scaled_expanded_function_ScaleFactorF, tmp42_Attention_test_attention_4d_gqa_scaled_expanded_function_QScaled);
      |                                                                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                         |
      |                                                                                         float *
/tmp/tmppzik46xq/model.c:952:84: note: expected ‘const float (* restrict)[9][4][8]’ but argument is of type ‘float *’
  952 | static inline void node42_mul(const float input0[restrict 2][9][4][8], const float input1[restrict 2][9][4][8], float output[restrict 2][9][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1288:91: warning: passing argument 2 of ‘node43_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1288 |     node43_mul(tmp41_Attention_test_attention_4d_gqa_scaled_expanded_function_KTranspose, tmp15_Attention_test_attention_4d_gqa_scaled_expanded_function_ScaleFactorF, tmp43_Attention_test_attention_4d_gqa_scaled_expanded_function_KScaled);
      |                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                           |
      |                                                                                           float *
/tmp/tmppzik46xq/model.c:972:84: note: expected ‘const float (* restrict)[9][8][6]’ but argument is of type ‘float *’
  972 | static inline void node43_mul(const float input0[restrict 2][9][8][6], const float input1[restrict 2][9][8][6], float output[restrict 2][9][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1289:163: warning: passing argument 3 of ‘node44_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1289 |     node44_matmul(tmp42_Attention_test_attention_4d_gqa_scaled_expanded_function_QScaled, tmp43_Attention_test_attention_4d_gqa_scaled_expanded_function_KScaled, tmp44_Attention_test_attention_4d_gqa_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                   |
      |                                                                                                                                                                   float (*)[6]
/tmp/tmppzik46xq/model.c:992:122: note: expected ‘float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
  992 | static inline void node44_matmul(const float input0[restrict 2][9][4][8], const float input1[restrict 2][9][8][6], float output[restrict 2][9][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1290:17: warning: passing argument 1 of ‘node45_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1290 |     node45_cast(tmp44_Attention_test_attention_4d_gqa_scaled_expanded_function_QKAttnWeight, tmp45_Attention_test_attention_4d_gqa_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmppzik46xq/model.c:1017:44: note: expected ‘const float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
 1017 | static inline void node45_cast(const float input0[restrict 2][9][4][6], float output[restrict 2][9][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1291:91: warning: passing argument 2 of ‘node46_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1291 |     node46_add(tmp45_Attention_test_attention_4d_gqa_scaled_expanded_function_QKAttnCast, tmp22_Attention_test_attention_4d_gqa_scaled_expanded_function_AttnBiasT, tmp46_Attention_test_attention_4d_gqa_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                           |
      |                                                                                           float (*)[6]
/tmp/tmppzik46xq/model.c:1037:84: note: expected ‘const float (* restrict)[9][4][6]’ but argument is of type ‘float (*)[6]’
 1037 | static inline void node46_add(const float input0[restrict 2][9][4][6], const float input1[restrict 2][9][4][6], float output[restrict 2][9][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1296:174: warning: passing argument 3 of ‘node51_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1296 |     node51_matmul(tmp50_Attention_test_attention_4d_gqa_scaled_expanded_function_SoftmaxOut, tmp40_Attention_test_attention_4d_gqa_scaled_expanded_function_VAttentionInput, tmp51_Attention_test_attention_4d_gqa_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                              |
      |                                                                                                                                                                              float (*)[8]
/tmp/tmppzik46xq/model.c:1156:122: note: expected ‘float (* restrict)[9][4][8]’ but argument is of type ‘float (*)[8]’
 1156 | static inline void node51_matmul(const float input0[restrict 2][9][4][6], const float input1[restrict 2][9][6][8], float output[restrict 2][9][4][8]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmppzik46xq/model.c:1297:21: warning: passing argument 1 of ‘node52_identity’ from incompatible pointer type [-Wincompatible-pointer-types]
 1297 |     node52_identity(tmp51_Attention_test_attention_4d_gqa_scaled_expanded_function_YPreReshape, Y);
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                     |
      |                     float (*)[8]
/tmp/tmppzik46xq/model.c:1180:48: note: expected ‘const float (* restrict)[9][4][8]’ but argument is of type ‘float (*)[8]’
 1180 | static inline void node52_identity(const float input0[restrict 2][9][4][8], float output[restrict 2][9][4][8]) {
      |                                    ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Out of tolerance (max ULP 3042080) | 1 | █ |
| Out of tolerance (max ULP 300) | 1 | █ |
| Failed to build testbench: /tmp/tmp_8ixz4q8/model.c: In function ‘node13_identity’:
/tmp/tmp_8ixz4q8/model.c:384:12: error: ‘i0’ undeclared (first use in this function)
  384 |     output[i0] = input0[i0];
      |            ^~
/tmp/tmp_8ixz4q8/model.c:384:12: note: each undeclared identifier is reported only once for each function it appears in
/tmp/tmp_8ixz4q8/model.c: In function ‘node14_sqrt’:
/tmp/tmp_8ixz4q8/model.c:396:12: error: ‘i0’ undeclared (first use in this function)
  396 |     output[i0] = ref_scalar_f32_sqrt(input0[i0]);
      |            ^~
/tmp/tmp_8ixz4q8/model.c: In function ‘node15_cast’:
/tmp/tmp_8ixz4q8/model.c:409:12: error: ‘i0’ undeclared (first use in this function)
  409 |     output[i0] = (float)input0[i0];
      |            ^~
/tmp/tmp_8ixz4q8/model.c: In function ‘model’:
/tmp/tmp_8ixz4q8/model.c:1287:85: warning: passing argument 2 of ‘node42_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1287 |     node42_mul(tmp3_Attention_test_attention_4d_scaled_expanded_function_QReshaped, tmp15_Attention_test_attention_4d_scaled_expanded_function_ScaleFactorF, tmp42_Attention_test_attention_4d_scaled_expanded_function_QScaled);
      |                                                                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                     |
      |                                                                                     float *
/tmp/tmp_8ixz4q8/model.c:952:84: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float *’
  952 | static inline void node42_mul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][4][8], float output[restrict 2][3][4][8]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1288:87: warning: passing argument 2 of ‘node43_mul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1288 |     node43_mul(tmp41_Attention_test_attention_4d_scaled_expanded_function_KTranspose, tmp15_Attention_test_attention_4d_scaled_expanded_function_ScaleFactorF, tmp43_Attention_test_attention_4d_scaled_expanded_function_KScaled);
      |                                                                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                       |
      |                                                                                       float *
/tmp/tmp_8ixz4q8/model.c:972:84: note: expected ‘const float (* restrict)[3][8][6]’ but argument is of type ‘float *’
  972 | static inline void node43_mul(const float input0[restrict 2][3][8][6], const float input1[restrict 2][3][8][6], float output[restrict 2][3][8][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1289:155: warning: passing argument 3 of ‘node44_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1289 |     node44_matmul(tmp42_Attention_test_attention_4d_scaled_expanded_function_QScaled, tmp43_Attention_test_attention_4d_scaled_expanded_function_KScaled, tmp44_Attention_test_attention_4d_scaled_expanded_function_QKAttnWeight);
      |                                                                                                                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                           |
      |                                                                                                                                                           float (*)[6]
/tmp/tmp_8ixz4q8/model.c:992:122: note: expected ‘float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
  992 | static inline void node44_matmul(const float input0[restrict 2][3][4][8], const float input1[restrict 2][3][8][6], float output[restrict 2][3][4][6]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1290:17: warning: passing argument 1 of ‘node45_cast’ from incompatible pointer type [-Wincompatible-pointer-types]
 1290 |     node45_cast(tmp44_Attention_test_attention_4d_scaled_expanded_function_QKAttnWeight, tmp45_Attention_test_attention_4d_scaled_expanded_function_QKAttnCast);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 |
      |                 float (*)[6]
/tmp/tmp_8ixz4q8/model.c:1017:44: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1017 | static inline void node45_cast(const float input0[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1291:87: warning: passing argument 2 of ‘node46_add’ from incompatible pointer type [-Wincompatible-pointer-types]
 1291 |     node46_add(tmp45_Attention_test_attention_4d_scaled_expanded_function_QKAttnCast, tmp22_Attention_test_attention_4d_scaled_expanded_function_AttnBiasT, tmp46_Attention_test_attention_4d_scaled_expanded_function_QKAttnWeightWithBias);
      |                                                                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                       |
      |                                                                                       float (*)[6]
/tmp/tmp_8ixz4q8/model.c:1037:84: note: expected ‘const float (* restrict)[3][4][6]’ but argument is of type ‘float (*)[6]’
 1037 | static inline void node46_add(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][4][6], float output[restrict 2][3][4][6]) {
      |                                                                        ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1296:166: warning: passing argument 3 of ‘node51_matmul’ from incompatible pointer type [-Wincompatible-pointer-types]
 1296 |     node51_matmul(tmp50_Attention_test_attention_4d_scaled_expanded_function_SoftmaxOut, tmp40_Attention_test_attention_4d_scaled_expanded_function_VAttentionInput, tmp51_Attention_test_attention_4d_scaled_expanded_function_YPreReshape);
      |                                                                                                                                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                                                                                      |
      |                                                                                                                                                                      float (*)[8]
/tmp/tmp_8ixz4q8/model.c:1156:122: note: expected ‘float (* restrict)[3][4][8]’ but argument is of type ‘float (*)[8]’
 1156 | static inline void node51_matmul(const float input0[restrict 2][3][4][6], const float input1[restrict 2][3][6][8], float output[restrict 2][3][4][8]) {
      |                                                                                                                    ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tmp_8ixz4q8/model.c:1297:21: warning: passing argument 1 of ‘node52_identity’ from incompatible pointer type [-Wincompatible-pointer-types]
 1297 |     node52_identity(tmp51_Attention_test_attention_4d_scaled_expanded_function_YPreReshape, Y);
      |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                     |
      |                     float (*)[8]
/tmp/tmp_8ixz4q8/model.c:1180:48: note: expected ‘const float (* restrict)[3][4][8]’ but argument is of type ‘float (*)[8]’
 1180 | static inline void node52_identity(const float input0[restrict 2][3][4][8], float output[restrict 2][3][4][8]) {
      |                                    ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~ | 1 | █ |
| Out of tolerance (max ULP 1902013) | 1 | █ |
| Out of tolerance (max ULP 1071538375) | 1 | █ |
| Out of tolerance (max ULP 85699935) | 1 | █ |
| Out of tolerance (max ULP 1073227119) | 1 | █ |
| Out of tolerance (max ULP 1067548513) | 1 | █ |
| Out of tolerance (max ULP 1072671123) | 1 | █ |
| Out of tolerance (max ULP 1068382978) | 1 | █ |
| Out of tolerance (max ULP 15953592) | 1 | █ |
| Failed to build testbench: /tmp/tmp9r4okvcg/model.c:74:195: error: ‘nan’ undeclared here (not in a function)
   74 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmp9r4okvcg/model.c:74:198: error: expected ‘}’ before numeric constant
   74 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmp9r4okvcg/model.c:73:46: note: to match this ‘{’
   73 | static const double input_testbench_data[] = {
      |                                              ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpazb86_5p/model.c:74:195: error: ‘nan’ undeclared here (not in a function)
   74 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmpazb86_5p/model.c:74:198: error: expected ‘}’ before numeric constant
   74 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmpazb86_5p/model.c:73:46: note: to match this ‘{’
   73 | static const double input_testbench_data[] = {
      |                                              ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpy_e7yvpz/model.c:74:229: error: ‘nan’ undeclared here (not in a function)
   74 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmpy_e7yvpz/model.c:74:232: error: expected ‘}’ before numeric constant
   74 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmpy_e7yvpz/model.c:73:48: note: to match this ‘{’
   73 | static const _Float16 input_testbench_data[] = {
      |                                                ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpwzy5mo2k/model.c:74:229: error: ‘nan’ undeclared here (not in a function)
   74 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmpwzy5mo2k/model.c:74:232: error: expected ‘}’ before numeric constant
   74 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmpwzy5mo2k/model.c:73:48: note: to match this ‘{’
   73 | static const _Float16 input_testbench_data[] = {
      |                                                ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpe9_373l7/model.c:74:136: error: ‘nan’ undeclared here (not in a function)
   74 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmpe9_373l7/model.c:74:139: error: expected ‘}’ before numeric constant
   74 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmpe9_373l7/model.c:73:45: note: to match this ‘{’
   73 | static const float input_testbench_data[] = {
      |                                             ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpakz_uda1/model.c:74:136: error: ‘nan’ undeclared here (not in a function)
   74 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmpakz_uda1/model.c:74:139: error: expected ‘}’ before numeric constant
   74 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmpakz_uda1/model.c:73:45: note: to match this ‘{’
   73 | static const float input_testbench_data[] = {
      |                                             ^ | 1 | █ |
| Failed to build testbench: /tmp/tmpqkm4h4a6/model.c:73:195: error: ‘nan’ undeclared here (not in a function)
   73 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmpqkm4h4a6/model.c:73:198: error: expected ‘}’ before numeric constant
   73 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmpqkm4h4a6/model.c:72:46: note: to match this ‘{’
   72 | static const double input_testbench_data[] = {
      |                                              ^
/tmp/tmpqkm4h4a6/model.c: In function ‘main’:
/tmp/tmpqkm4h4a6/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpqkm4h4a6/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmp_fxq0raf/model.c:73:195: error: ‘nan’ undeclared here (not in a function)
   73 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmp_fxq0raf/model.c:73:198: error: expected ‘}’ before numeric constant
   73 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmp_fxq0raf/model.c:72:46: note: to match this ‘{’
   72 | static const double input_testbench_data[] = {
      |                                              ^
/tmp/tmp_fxq0raf/model.c: In function ‘main’:
/tmp/tmp_fxq0raf/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmp_fxq0raf/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpwg2h7qs1/model.c:75:195: error: ‘nan’ undeclared here (not in a function)
   75 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmpwg2h7qs1/model.c:75:198: error: expected ‘}’ before numeric constant
   75 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmpwg2h7qs1/model.c:74:46: note: to match this ‘{’
   74 | static const double input_testbench_data[] = {
      |                                              ^
/tmp/tmpwg2h7qs1/model.c: In function ‘main’:
/tmp/tmpwg2h7qs1/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpwg2h7qs1/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmp22mywjen/model.c:75:195: error: ‘nan’ undeclared here (not in a function)
   75 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                   ^~~
/tmp/tmp22mywjen/model.c:75:198: error: expected ‘}’ before numeric constant
   75 |     0.47892546653747559,    0.48033666610717773,    0.49968487024307251,    0.81910544633865356,    0.47031247615814209,    0.8164680004119873,    0.21087194979190826,    0.7229037880897522,    nan.0,    inf.0,    inf.0,    -inf.0};
      |                                                                                                                                                                                                      ^~
/tmp/tmp22mywjen/model.c:74:46: note: to match this ‘{’
   74 | static const double input_testbench_data[] = {
      |                                              ^
/tmp/tmp22mywjen/model.c: In function ‘main’:
/tmp/tmp22mywjen/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmp22mywjen/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpl9cbzm7p/model.c:73:229: error: ‘nan’ undeclared here (not in a function)
   73 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmpl9cbzm7p/model.c:73:232: error: expected ‘}’ before numeric constant
   73 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmpl9cbzm7p/model.c:72:48: note: to match this ‘{’
   72 | static const _Float16 input_testbench_data[] = {
      |                                                ^
/tmp/tmpl9cbzm7p/model.c: In function ‘main’:
/tmp/tmpl9cbzm7p/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpl9cbzm7p/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpz7tcp6kh/model.c:75:229: error: ‘nan’ undeclared here (not in a function)
   75 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmpz7tcp6kh/model.c:75:232: error: expected ‘}’ before numeric constant
   75 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmpz7tcp6kh/model.c:74:48: note: to match this ‘{’
   74 | static const _Float16 input_testbench_data[] = {
      |                                                ^
/tmp/tmpz7tcp6kh/model.c: In function ‘main’:
/tmp/tmpz7tcp6kh/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpz7tcp6kh/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmp9yvf7sam/model.c:73:229: error: ‘nan’ undeclared here (not in a function)
   73 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmp9yvf7sam/model.c:73:232: error: expected ‘}’ before numeric constant
   73 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmp9yvf7sam/model.c:72:48: note: to match this ‘{’
   72 | static const _Float16 input_testbench_data[] = {
      |                                                ^
/tmp/tmp9yvf7sam/model.c: In function ‘main’:
/tmp/tmp9yvf7sam/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmp9yvf7sam/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpndv1sgod/model.c:75:229: error: ‘nan’ undeclared here (not in a function)
   75 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                     ^~~
/tmp/tmpndv1sgod/model.c:75:232: error: expected ‘}’ before numeric constant
   75 |     (_Float16)0.479003906f,    (_Float16)0.480224609f,    (_Float16)0.499755859f,    (_Float16)0.819335938f,    (_Float16)0.470214844f,    (_Float16)0.81640625f,    (_Float16)0.21081543f,    (_Float16)0.723144531f,    (_Float16)nan.0f,    (_Float16)inf.0f,    (_Float16)inf.0f,    (_Float16)-inf.0f};
      |                                                                                                                                                                                                                                        ^~~
/tmp/tmpndv1sgod/model.c:74:48: note: to match this ‘{’
   74 | static const _Float16 input_testbench_data[] = {
      |                                                ^
/tmp/tmpndv1sgod/model.c: In function ‘main’:
/tmp/tmpndv1sgod/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpndv1sgod/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpmtzhfnme/model.c:73:136: error: ‘nan’ undeclared here (not in a function)
   73 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmpmtzhfnme/model.c:73:139: error: expected ‘}’ before numeric constant
   73 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmpmtzhfnme/model.c:72:45: note: to match this ‘{’
   72 | static const float input_testbench_data[] = {
      |                                             ^
/tmp/tmpmtzhfnme/model.c: In function ‘main’:
/tmp/tmpmtzhfnme/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpmtzhfnme/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmp65uuw62l/model.c:75:136: error: ‘nan’ undeclared here (not in a function)
   75 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmp65uuw62l/model.c:75:139: error: expected ‘}’ before numeric constant
   75 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmp65uuw62l/model.c:74:45: note: to match this ‘{’
   74 | static const float input_testbench_data[] = {
      |                                             ^
/tmp/tmp65uuw62l/model.c: In function ‘main’:
/tmp/tmp65uuw62l/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmp65uuw62l/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpvzxymdlt/model.c:73:136: error: ‘nan’ undeclared here (not in a function)
   73 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmpvzxymdlt/model.c:73:139: error: expected ‘}’ before numeric constant
   73 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmpvzxymdlt/model.c:72:45: note: to match this ‘{’
   72 | static const float input_testbench_data[] = {
      |                                             ^
/tmp/tmpvzxymdlt/model.c: In function ‘main’:
/tmp/tmpvzxymdlt/model.c:85:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   85 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpvzxymdlt/model.c:85:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Failed to build testbench: /tmp/tmpunc1mvn6/model.c:75:136: error: ‘nan’ undeclared here (not in a function)
   75 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                        ^~~
/tmp/tmpunc1mvn6/model.c:75:139: error: expected ‘}’ before numeric constant
   75 |     0.478925467f,    0.480336666f,    0.49968487f,    0.819105446f,    0.470312476f,    0.816468f,    0.21087195f,    0.722903788f,    nan.0f,    inf.0f,    inf.0f,    -inf.0f};
      |                                                                                                                                           ^~~
/tmp/tmpunc1mvn6/model.c:74:45: note: to match this ‘{’
   74 | static const float input_testbench_data[] = {
      |                                             ^
/tmp/tmpunc1mvn6/model.c: In function ‘main’:
/tmp/tmpunc1mvn6/model.c:87:20: error: ‘like_testbench_data’ undeclared (first use in this function); did you mean ‘input_testbench_data’?
   87 |         like[i0] = like_testbench_data[i0];
      |                    ^~~~~~~~~~~~~~~~~~~
      |                    input_testbench_data
/tmp/tmpunc1mvn6/model.c:87:20: note: each undeclared identifier is reported only once for each function it appears in | 1 | █ |
| Out of tolerance (max ULP 2827146620) | 1 | █ |
| Out of tolerance (max ULP 3079899563) | 1 | █ |
| Out of tolerance (max ULP 2947284595) | 1 | █ |
| Out of tolerance (max ULP 2134203891) | 1 | █ |
| Out of tolerance (max ULP 2965007219) | 1 | █ |
| Out of tolerance (max ULP 2763460791) | 1 | █ |
| Out of tolerance (max ULP 1069384066) | 1 | █ |
| Out of tolerance (max ULP 1072630820) | 1 | █ |
| Out of tolerance (max ULP 2130706433) | 1 | █ |
| Out of tolerance (max ULP 1084227584) | 1 | █ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Unsupported op ScatterND | 4 | ██████████████████████████████ |
| Unsupported LSTM direction b'*' | 2 | ███████████████ |
| Unsupported op QLinearAdd | 2 | ███████████████ |
| Unsupported op QLinearMul | 2 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ████████ |
| Out of tolerance (max ULP 591626278) | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |
