// softmax_v2.cu — GPU SIMT softmax with K cols/thread (thread coarsening).
// Mirrors the per-thread K-element reduction shape of DICE softmax_acc_v2,
// so the speedup numbers compare like-for-like.
//
// BLOCK_SIZE × K = COLS. We sweep K via the K macro at compile time.
//   K=4  → BLOCK_SIZE=128
//   K=8  → BLOCK_SIZE=64
//   K=16 → BLOCK_SIZE=32  (single warp; SMEM-tree degenerates to warp scan)
//
// Phase 1: per-thread local fmax over K strided cols, then SMEM tree max.
// Phase 2: per-thread local exp+sum over K cols, then SMEM tree sum.
// Phase 3: per-thread strided stores of exp(x-m)/s.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef K
#define K 4
#endif
#ifndef COLS_DEFAULT
#define COLS_DEFAULT 512
#endif
#define BLOCK_SIZE (COLS_DEFAULT / K)

__global__ void softmax_v2_kernel(const float *x, float *y, int COLS)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t base = (size_t)row * COLS + tid;

    // K strided loads
    float v[K];
    #pragma unroll
    for (int i = 0; i < K; i++) v[i] = x[base + i * BLOCK_SIZE];

    // Phase 1: local fmax then SMEM tree
    float local_max = v[0];
    #pragma unroll
    for (int i = 1; i < K; i++) local_max = fmaxf(local_max, v[i]);
    smem[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float m = smem[0];
    __syncthreads();

    // Phase 2: local exp+sum then SMEM tree
    float e[K];
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; i++) { e[i] = __expf(v[i] - m); local_sum += e[i]; }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float s_row = smem[0];

    // Phase 3: strided stores
    #pragma unroll
    for (int i = 0; i < K; i++) y[base + i * BLOCK_SIZE] = e[i] / s_row;
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
    int COLS = (argc > 2) ? atoi(argv[2]) : COLS_DEFAULT;
    if (COLS != BLOCK_SIZE * K) {
        fprintf(stderr, "COLS (%d) must equal BLOCK_SIZE*K (%d*%d=%d)\n",
                COLS, BLOCK_SIZE, K, BLOCK_SIZE * K);
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
    softmax_v2_kernel<<<ROWS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0; size_t first_idx = 0;
    for (size_t i = 0; i < N; i++) {
        if (!approx_eq(h_y[i], h_ref[i])) { if (mismatches == 0) first_idx = i; mismatches++; }
    }
    if (mismatches == 0)
        printf("softmax_v2(K=%d): CPU and GPU results match (ROWS=%d, COLS=%d, BLOCK=%d).\n",
               K, ROWS, COLS, BLOCK_SIZE);
    else
        printf("softmax_v2(K=%d): MISMATCH (%d at %zu: CPU=%f, GPU=%f).\n",
               K, mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
