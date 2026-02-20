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
void model(const float a[restrict 2][3][4], const float b[restrict 2][3][4], float out[restrict 2][3][4]);

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




__attribute__((weak, noinline)) void timer_start(void) {}
__attribute__((weak, noinline)) void timer_stop(void) {}


static void testbench_init_constant_input(void) {
    (void)0;
}

static int testbench_read_input_file(
const char *input_path, float a[2][3][4], float b[2][3][4]) {
    FILE *input_file = fopen(input_path, "rb");
    if (!input_file) {
        fprintf(stderr, "Failed to open input file: %s\n", input_path);
        return 1;
    }
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (fread(&a[i0][i1][i2], sizeof(float), 1, input_file) != 1) {
                    fprintf(stderr, "Failed to read input a\n");
                    fclose(input_file);
                    return 1;
                }
            }
        }
    }
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (fread(&b[i0][i1][i2], sizeof(float), 1, input_file) != 1) {
                    fprintf(stderr, "Failed to read input b\n");
                    fclose(input_file);
                    return 1;
                }
            }
        }
    }

    fclose(input_file);
    return 0;
}

static void testbench_fill_random_input(float a[2][3][4], float b[2][3][4]) {
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                a[i0][i1][i2] = rng_next_float();
            }
        }
    }
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                b[i0][i1][i2] = rng_next_float();
            }
        }
    }
}

static void testbench_print_json(
float a[2][3][4],
float b[2][3][4],
float out[2][3][4]) {
    printf("{\n  \"inputs\": {\n");
    printf("    \"a\": {\"shape\": [2,3,4], \"data\":\n      ");
    printf("[");
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        if (i0) {
            printf(",\n      ");
        }
        printf("[");
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            if (i1) {
                printf(",");
            }
            printf("[");
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (i2) {
                    printf(",");
                }
                printf("\"%a\"", (double)a[i0][i1][i2]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]}");
    printf(",\n");
    printf("    \"b\": {\"shape\": [2,3,4], \"data\":\n      ");
    printf("[");
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        if (i0) {
            printf(",\n      ");
        }
        printf("[");
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            if (i1) {
                printf(",");
            }
            printf("[");
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (i2) {
                    printf(",");
                }
                printf("\"%a\"", (double)b[i0][i1][i2]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]}");

    printf("\n  },\n  \"outputs\": {\n");
    printf("    \"out\": {\"shape\": [2,3,4], \"data\":\n      ");
    printf("[");
    for (idx_t i0 = 0; i0 < 2; ++i0) {
        if (i0) {
            printf(",\n      ");
        }
        printf("[");
        for (idx_t i1 = 0; i1 < 3; ++i1) {
            if (i1) {
                printf(",");
            }
            printf("[");
            for (idx_t i2 = 0; i2 < 4; ++i2) {
                if (i2) {
                    printf(",");
                }
                printf("\"%a\"", (double)out[i0][i1][i2]);
            }
            printf("]");
        }
        printf("]");
    }
    printf("]}");
    printf("\n  }\n}\n");
}

static int testbench_run(const char *input_path) {

    float a[2][3][4];
    float b[2][3][4];

    testbench_init_constant_input();
    if (input_path) {
        if (testbench_read_input_file(input_path, a, b) != 0) {
            return 1;
        }
    } else {
        testbench_fill_random_input(a, b);
    }

    float out[2][3][4];

    if (!model_load("model.bin")) {
        return 1;
    }

    timer_start();
    model(a, b, out);
    timer_stop();

    testbench_print_json(a, b, out);
    return 0;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;

    if (argc > 1) {
        input_path = argv[1];
    }

    return testbench_run(input_path);
}
