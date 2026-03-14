/* Testbench (separate translation unit). */
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>

#ifndef idx_t
#define idx_t int32_t
#endif
#ifndef EMX_STRING_MAX_LEN
#define EMX_STRING_MAX_LEN 256
#endif
#ifndef EMX_SEQUENCE_MAX_LEN
#define EMX_SEQUENCE_MAX_LEN 32
#endif

_Bool model_load(const char *path);
void model(const float theta[restrict 2][3][4], const int64_t size[restrict 5], float grid[restrict 2][4][5][6][3]);

static uint64_t rng_state = 0x243f6a8885a308d3ull;

static uint64_t rng_next_u64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545f4914f6cdd1dull;
}

static float rng_next_float(void) {
    return (float)((double)rng_next_u64() * (1.0 / 18446744073709551616.0));
}


static int64_t rng_next_i64(void) {
    return (int64_t)rng_next_u64();
}


__attribute__((weak, noinline)) void timer_start(void) {}
__attribute__((weak, noinline)) void timer_stop(void) {}


static void testbench_init_constant_input(void) {
    (void)0;
}

static int testbench_read_input_file(
const char *input_path, float theta[2][3][4], int64_t size[5]) {
    FILE *input_file = fopen(input_path, "rb");
    if (!input_file) {
        fprintf(stderr, "Failed to open input file: %s\n", input_path);
        return 1;
    }
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (fread(&theta[i0][i1][i2], sizeof(float), 1, input_file) != 1) {
                    fprintf(stderr, "Failed to read input theta\n");
                    fclose(input_file);
                    return 1;
                }
            }
        }
    }
    for (idx_t i0 = 0; i0 < 5; ++i0) {
        if (fread(&size[i0], sizeof(int64_t), 1, input_file) != 1) {
            fprintf(stderr, "Failed to read input size\n");
            fclose(input_file);
            return 1;
        }
    }

    fclose(input_file);
    return 0;
}

static void testbench_fill_random_input(float theta[2][3][4], int64_t size[5]) {
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                theta[i0][i1][i2] = rng_next_float();
            }
        }
    }
    for (idx_t i0 = 0; i0 < 5; ++i0) {
        size[i0] = (int64_t)rng_next_i64();
    }
}

static void testbench_print_json(
float grid[2][4][5][6][3]) {
    printf("{\n  \"outputs\": {\n");
    printf("    \"grid\": {\"shape\": [2,4,5,6,3], \"data\":\n      ");
    printf("[");
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        if (i0) {
            printf(",\n      ");
        }
        printf("[");
        for (idx_t i1 = 0; i1 < 4; ++i1) {
            if (i1) {
                printf(",");
            }
            printf("[");
            for (idx_t i2 = 0; i2 < 5; ++i2) {
                if (i2) {
                    printf(",");
                }
                printf("[");
                for (idx_t i3 = 0; i3 < 6; ++i3) {
                    if (i3) {
                        printf(",");
                    }
                    printf("[");
                    for (idx_t i4 = 0; i4 < 3; ++i4) {
                        if (i4) {
                            printf(",");
                        }
                        printf("\"%a\"", (double)grid[i0][i1][i2][i3][i4]);
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }
    printf("]}");
    printf("\n  }\n}\n");
}

static int testbench_run(const char *input_path) {

    float theta[2][3][4];
    int64_t size[5];

    testbench_init_constant_input();
    if (input_path) {
        if (testbench_read_input_file(input_path, theta, size) != 0) {
            return 1;
        }
    } else {
        testbench_fill_random_input(theta, size);
    }

    float grid[2][4][5][6][3];

    if (!model_load("model.bin")) {
        return 1;
    }

    timer_start();
    model(theta, size, grid);
    timer_stop();

    testbench_print_json(grid);
    return 0;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;

    if (argc > 1) {
        input_path = argv[1];
    }

    return testbench_run(input_path);
}
