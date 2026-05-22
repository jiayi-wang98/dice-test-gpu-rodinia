// softmax_acc.cu — row-wise numerically-stable softmax (DICE+IFF
// variant).  Same three-pass shape as the GPU baseline, but each
// per-row tree reduction is replaced by a single IFF dispatch:
//   Pass 1: dice_cta_acc_max (MAX-mode IFF)
//   Pass 2: dice_cta_acc_add (ADD-mode IFF) on exp(x - m)
//   Pass 3: per-thread divide
// This is the first benchmark that exercises *both* IFF op modes
// inside one kernel.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

__global__ void softmax_acc_kernel(const float *x, float *y, int COLS)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Slot 0 holds the running max in sortable-int encoding (so we can
    // use a single atomicMax instead of a divergence-prone CAS-loop).
    // The init value is DICE_ACC_MAX_INIT_F (== INT_MIN, == sortable
    // encoding of -FLT_MAX).
    // Slot 1 holds the running sum (0.0f identity, float-encoded).
    //
    if (tid == 0) {
        ((int   *)__dice_acc_slots)[0] = DICE_ACC_MAX_INIT_F;
        ((float *)__dice_acc_slots)[1] = 0.0f;
    }
    __syncthreads();

    float xi = x[(size_t)row * COLS + tid];

    // Pass 1: MAX-mode IFF over the row.
    float m_running;
    dice_cta_acc_max(m_running, xi);
    __syncthreads();
    // Decode the max slot from sortable-int back to float.
    float m = dice_sortable_to_float(((int *)__dice_acc_slots)[0]);

    // Pass 2: ADD-mode IFF over exp(xi - m).
    float e = __expf(xi - m);
    float s_running;
    dice_cta_acc_add(s_running, e);
    __syncthreads();
    float s_row = ((float *)__dice_acc_slots)[1];

    // Pass 3: divide.
    y[(size_t)row * COLS + tid] = e / s_row;
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
    int COLS = (argc > 2) ? atoi(argv[2]) : BLOCK_SIZE;
    if (COLS != BLOCK_SIZE) {
        fprintf(stderr, "COLS must equal BLOCK_SIZE (%d) for this kernel\n",
                BLOCK_SIZE);
        return 1;
    }

    size_t N = (size_t)ROWS * COLS;
    float *h_x   = (float *)malloc(N * sizeof(float));
    float *h_y   = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    srand(11);
    for (size_t i = 0; i < N; i++) {
        h_x[i] = (((rand() & 0xFFFF) / 65535.0f) - 0.5f) * 8.0f;  // [-4, 4]
    }
    softmax_cpu(h_x, h_ref, ROWS, COLS);

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax_acc_kernel<<<ROWS, BLOCK_SIZE>>>(d_x, d_y, COLS);
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

    if (mismatches == 0) {
        printf("softmax_acc: CPU and GPU results match (ROWS=%d, COLS=%d).\n",
               ROWS, COLS);
    } else {
        printf("softmax_acc: MISMATCH (ROWS=%d, COLS=%d, %d mismatches; first at %zu: CPU=%f, GPU=%f).\n",
               ROWS, COLS, mismatches, first_idx,
               h_ref[first_idx], h_y[first_idx]);
    }

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
