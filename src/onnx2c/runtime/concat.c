#include "concat.h"

#include <string.h>

void onnx2c_concat(
    const void *const *inputs,
    const size_t *axis_sizes,
    size_t input_count,
    size_t outer,
    size_t inner,
    size_t element_size,
    void *output
) {
    size_t concat_axis = 0;
    for (size_t idx = 0; idx < input_count; ++idx) {
        concat_axis += axis_sizes[idx];
    }
    if (concat_axis == 0 || element_size == 0) {
        return;
    }

    unsigned char *output_bytes = (unsigned char *)output;
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        size_t output_offset = outer_idx * concat_axis * inner;
        size_t axis_offset = 0;
        for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
            size_t axis = axis_sizes[input_idx];
            size_t copy_elems = axis * inner;
            const unsigned char *input_bytes =
                (const unsigned char *)inputs[input_idx];
            size_t input_offset = outer_idx * copy_elems;
            memcpy(
                output_bytes + (output_offset + axis_offset) * element_size,
                input_bytes + input_offset * element_size,
                copy_elems * element_size
            );
            axis_offset += copy_elems;
        }
    }
}
