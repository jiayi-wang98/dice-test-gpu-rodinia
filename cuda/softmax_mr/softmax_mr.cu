// softmax_mr.cu — multi-row-per-CTA DICE softmax.
//
// Keeps BLOCK_SIZE=512 (full CGRA-core thread budget) and packs
// ROWS_PER_CTA rows into one CTA.  Each row gets its OWN accumulator
// slot pair, so the M rows reduce as M parallel acc-PE streams.
//
//   ROWS_PER_CTA = 8  -> 64 threads/row, K=8 cols/thread, 16 slots
//   ROWS_PER_CTA = 16 -> 32 threads/row, K=16 cols/thread, 32 slots
//
// Slot layout: slots[0 .. M-1]   = per-row MAX (sortable-int)
//              slots[M .. 2M-1]  = per-row SUM (float)
//
// Each thread:
//   row_in_cta = tid / THREADS_PER_ROW   (which row inside this CTA)
//   lane       = tid % THREADS_PER_ROW   (position within the row)
//   handles K strided cols: base + k*THREADS_PER_ROW, k in [0,K)
// Strided so the THREADS_PER_ROW threads of a row group coalesce.
//
// On DICE the acc-PE registers are born initialized (MAX=-inf, ADD=0);
// the explicit init below is only needed for the GPU-SIMT reference
// build (shared memory is uninitialized on real GPUs).  The DICE .pptx
// folds the init in as a free, zero-resource prologue.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../dice_test/dice_atomics.h"

#ifndef COLS_DEFAULT
#define COLS_DEFAULT 512
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef ROWS_PER_CTA
#define ROWS_PER_CTA 8
#endif
#define THREADS_PER_ROW (BLOCK_SIZE / ROWS_PER_CTA)
#define K (COLS_DEFAULT / THREADS_PER_ROW)

__global__ void softmax_mr_kernel(const float *x, float *y, int COLS)
{
    int tid  = threadIdx.x;
    int ric  = tid / THREADS_PER_ROW;             // row-in-CTA  [0, ROWS_PER_CTA)
    int lane = tid % THREADS_PER_ROW;             // lane within row [0, TPR)
    int grow = blockIdx.x * ROWS_PER_CTA + ric;   // global row
    int mslot = ric;                              // MAX slot for this row
    int sslot = ROWS_PER_CTA + ric;               // SUM slot for this row

    // GPU-reference init (DICE fabric does this for free): first 2*M
    // threads clear the 2*M slots.
    if (tid < ROWS_PER_CTA) {
        ((int   *)__dice_acc_slots)[tid]                = DICE_ACC_MAX_INIT_F;
        ((float *)__dice_acc_slots)[ROWS_PER_CTA + tid] = 0.0f;
    }
    __syncthreads();

    // K strided loads (coalesced across the THREADS_PER_ROW row group).
    size_t base = (size_t)grow * COLS + lane;
    float v[K];
    #pragma unroll
    for (int k = 0; k < K; k++) v[k] = x[base + (size_t)k * THREADS_PER_ROW];

    // Pass 1: per-thread local max, then per-row acc-PE MAX.
    float lmax = v[0];
    #pragma unroll
    for (int k = 1; k < K; k++) lmax = fmaxf(lmax, v[k]);
    atomicMax((int *)&__dice_acc_slots[mslot], dice_float_to_sortable(lmax));
    __syncthreads();
    float m = dice_sortable_to_float(((int *)__dice_acc_slots)[mslot]);

    // Pass 2: per-thread exp + local sum, then per-row acc-PE ADD.
    float e[K];
    float lsum = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; k++) { e[k] = __expf(v[k] - m); lsum += e[k]; }
    atomicAdd((float *)&__dice_acc_slots[sslot], lsum);
    __syncthreads();
    float s = ((float *)__dice_acc_slots)[sslot];

    // Pass 3: normalize + strided store.
    float inv = 1.0f / s;
    #pragma unroll
    for (int k = 0; k < K; k++) y[base + (size_t)k * THREADS_PER_ROW] = e[k] * inv;
}

// -----------------------------------------------------------------------------

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
    int ROWS = (argc > 1) ? atoi(argv[1]) : 4096;
    int COLS = (argc > 2) ? atoi(argv[2]) : COLS_DEFAULT;
    if (COLS != COLS_DEFAULT) {
        fprintf(stderr, "COLS (%d) must equal COLS_DEFAULT (%d)\n", COLS, COLS_DEFAULT);
        return 1;
    }
    if (ROWS % ROWS_PER_CTA != 0) {
        fprintf(stderr, "ROWS (%d) must be a multiple of ROWS_PER_CTA (%d)\n",
                ROWS, ROWS_PER_CTA);
        return 1;
    }

    size_t N = (size_t)ROWS * COLS;
    float *h_x   = (float *)malloc(N * sizeof(float));
    float *h_y   = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));

    srand(11);
    for (size_t i = 0; i < N; i++)
        h_x[i] = (((rand() & 0xFFFF) / 65535.0f) - 0.5f) * 8.0f;
    softmax_cpu(h_x, h_ref, ROWS, COLS);

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int nctas = ROWS / ROWS_PER_CTA;
    softmax_mr_kernel<<<nctas, BLOCK_SIZE>>>(d_x, d_y, COLS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    size_t first_idx = 0;
    for (size_t i = 0; i < N; i++) {
        if (!approx_eq(h_y[i], h_ref[i])) { if (mismatches == 0) first_idx = i; mismatches++; }
    }
    printf("softmax_mr: ROWS=%d, COLS=%d, BLOCK=%d, ROWS_PER_CTA=%d, TPR=%d, K=%d, nCTA=%d\n",
           ROWS, COLS, BLOCK_SIZE, ROWS_PER_CTA, THREADS_PER_ROW, K, nctas);
    if (mismatches == 0)
        printf("softmax_mr: CPU and GPU results match.\n");
    else
        printf("softmax_mr: MISMATCH (%d at %zu: CPU=%f, GPU=%f).\n",
               mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
