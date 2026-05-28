// softmax_acc_v2.cu — per-thread K-column local reduction + single
// per-row acc-PE chain (one MAX, one ADD).
//
// v1 (softmax_acc.cu): BLOCK_SIZE=COLS=512, 1 col/thread, 1 acc.max +
// 1 acc.add per thread. Total per row: 512 acc.max + 512 acc.add.
//
// v2: BLOCK_SIZE=128, COLS=512, K=4 cols/thread. Each thread does
// 4 local max ops + 1 acc.max, and 4 local sum ops + 1 acc.add.
// Total per row: 128 acc.max + 128 acc.add (4× fewer dispatches).
// Memory access count unchanged (still 512 loads + 512 stores per row).
//
// Load/store pattern is STRIDED across threads (thread t hits cols
// t, t+128, t+256, t+384) — gives one coalesced cache-line transaction
// per warp per iteration, matching the GPU baseline's access count.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 4
#endif

// fmaxf helper used by the local max reduce.
__device__ __forceinline__ float fmax4(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

__global__ void softmax_acc_v2_kernel(const float *x, float *y, int COLS)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // No explicit slot init / barrier: on DICE the accumulator-PE registers
    // are born initialized at CTA dispatch (MAX slot = -inf sentinel, ADD
    // slot = 0). The init is a fabric property, not user work.

    // K=4 strided loads per thread
    size_t base = (size_t)row * COLS + tid;
    float x0 = x[base + 0 * BLOCK_SIZE];
    float x1 = x[base + 1 * BLOCK_SIZE];
    float x2 = x[base + 2 * BLOCK_SIZE];
    float x3 = x[base + 3 * BLOCK_SIZE];

    // Pass 1: per-thread local max, then acc-PE MAX over all threads.
    float local_max = fmax4(x0, x1, x2, x3);
    float m_running;
    dice_cta_acc_max(m_running, local_max);
    __syncthreads();
    float m = dice_sortable_to_float(((int *)__dice_acc_slots)[0]);

    // Pass 2: per-thread local sum of exp(xi - m), then acc-PE ADD.
    float e0 = __expf(x0 - m);
    float e1 = __expf(x1 - m);
    float e2 = __expf(x2 - m);
    float e3 = __expf(x3 - m);
    float local_sum = e0 + e1 + e2 + e3;
    float s_running;
    dice_cta_acc_add(s_running, local_sum);
    __syncthreads();
    float s_row = ((float *)__dice_acc_slots)[1];

    // Pass 3: per-thread strided stores
    y[base + 0 * BLOCK_SIZE] = e0 / s_row;
    y[base + 1 * BLOCK_SIZE] = e1 / s_row;
    y[base + 2 * BLOCK_SIZE] = e2 / s_row;
    y[base + 3 * BLOCK_SIZE] = e3 / s_row;
}

void softmax_cpu(const float *x, float *y, int ROWS, int COLS)
{
    for (int r = 0; r < ROWS; r++) {
        const float *xp = x + (size_t)r * COLS;
        float       *yp = y + (size_t)r * COLS;
        float m = -FLT_MAX;
        for (int c = 0; c < COLS; c++) if (xp[c] > m) m = xp[c];
        double s = 0.0;
        for (int c = 0; c < COLS; c++) s += expf(xp[c] - m);
        float inv = (float)(1.0 / s);
        for (int c = 0; c < COLS; c++) yp[c] = expf(xp[c] - m) * inv;
    }
}

static int approx_eq(float a, float b)
{
    float diff  = fabsf(a - b);
    float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return diff <= 1e-3f * scale;
}

int main(int argc, char **argv)
{
    int ROWS = (argc > 1) ? atoi(argv[1]) : 2048;
    int COLS = (argc > 2) ? atoi(argv[2]) : (BLOCK_SIZE * ELEMS_PER_THREAD);
    if (COLS != BLOCK_SIZE * ELEMS_PER_THREAD) {
        fprintf(stderr, "COLS (%d) must equal BLOCK_SIZE*ELEMS_PER_THREAD (%d)\n",
                COLS, BLOCK_SIZE * ELEMS_PER_THREAD);
        return 1;
    }

    size_t N = (size_t)ROWS * COLS;
    float *h_x   = (float *)malloc(N * sizeof(float));
    float *h_y   = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    srand(11);
    for (size_t i = 0; i < N; i++) {
        h_x[i] = (((rand() & 0xFFFF) / 65535.0f) - 0.5f) * 8.0f;
    }
    softmax_cpu(h_x, h_ref, ROWS, COLS);

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax_acc_v2_kernel<<<ROWS, BLOCK_SIZE>>>(d_x, d_y, COLS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    size_t first_idx = 0;
    for (size_t i = 0; i < N; i++) {
        if (!approx_eq(h_y[i], h_ref[i])) {
            if (mismatches == 0) first_idx = i;
            mismatches++;
        }
    }

    printf("softmax_acc_v2: ROWS=%d, COLS=%d, BLOCK=%d, K=%d\n",
           ROWS, COLS, BLOCK_SIZE, ELEMS_PER_THREAD);
    if (mismatches == 0) {
        printf("softmax_acc_v2: CPU and GPU results match.\n");
    } else {
        printf("softmax_acc_v2: MISMATCH (%d mismatches; first at %zu: CPU=%f, GPU=%f).\n",
               mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);
    }

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
