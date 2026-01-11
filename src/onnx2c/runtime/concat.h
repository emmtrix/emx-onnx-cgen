#ifndef ONNX2C_RUNTIME_CONCAT_H
#define ONNX2C_RUNTIME_CONCAT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void onnx2c_concat(
    const void *const *inputs,
    const size_t *axis_sizes,
    size_t input_count,
    size_t outer,
    size_t inner,
    size_t element_size,
    void *output
);

#ifdef __cplusplus
}
#endif

#endif
