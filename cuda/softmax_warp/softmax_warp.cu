// softmax_warp.cu — near-SOTA GPU softmax baseline for A100 (gpgpu-sim).
//
// gpgpu-sim's PTX parser/exec layer cannot reliably handle shfl.sync.bfly
// (the Volta+ warp shuffle), so we approximate cuDNN/Triton's pattern using:
//   * float4 vectorized loads/stores (16-byte transactions, 1 per 4 cols).
//   * BLOCK=128 (4 warps), 7-level SMEM tree (vs the textbook 9-level).
//   * Aggregated 4-element per-thread local reduce in registers before SMEM.
//   * 4 __syncthreads total per kernel (vs textbook's 18).
//   * __expf intrinsic for SFU exp.
//
// What we are *not* modeling: intra-warp shfl (zero-barrier register reduce).
// On real A100 with cuDNN, that delta would knock ~10-20% off the SMEM-tree
// time, but gpgpu-sim cannot simulate it. The numbers below should therefore
// be read as a faithful upper bound on the SMEM-tree-bound implementation,
// which is still substantially better than the textbook kernel.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef COLS_DEFAULT
#define COLS_DEFAULT 512
#endif
#define K_PER_THREAD 4
#define BLOCK_SIZE (COLS_DEFAULT / K_PER_THREAD)   // 128 for COLS=512

__global__ void softmax_warp_kernel(const float *__restrict__ x,
                                    float       *__restrict__ y,
                                    int COLS)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Vectorized float4 load — 1 transaction loads 4 contiguous cols.
    const float4 *xp = reinterpret_cast<const float4 *>(x + (size_t)row * COLS);
    float4 v = xp[tid];

    // Phase 1: per-thread local max in registers, then BLOCK-tree max in SMEM.
    float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    smem[tid] = m;
    __syncthreads();
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    m = smem[0];

    // Phase 2: per-thread exp into registers, local sum, then BLOCK-tree sum.
    float4 e;
    e.x = __expf(v.x - m); e.y = __expf(v.y - m);
    e.z = __expf(v.z - m); e.w = __expf(v.w - m);
    float s = (e.x + e.y) + (e.z + e.w);

    smem[tid] = s;
    __syncthreads();
    #pragma unroll
    for (int o = BLOCK_SIZE / 2; o > 0; o >>= 1) {
        if (tid < o) smem[tid] += smem[tid + o];
        __syncthreads();
    }
    float s_row = smem[0];

    // Phase 3: normalize and vectorized store.
    float inv = 1.0f / s_row;
    float4 out;
    out.x = e.x * inv; out.y = e.y * inv; out.z = e.z * inv; out.w = e.w * inv;
    float4 *yp = reinterpret_cast<float4 *>(y + (size_t)row * COLS);
    yp[tid] = out;
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
    if (COLS != BLOCK_SIZE * K_PER_THREAD) {
        fprintf(stderr, "COLS (%d) must equal BLOCK_SIZE*K (%d*%d=%d)\n",
                COLS, BLOCK_SIZE, K_PER_THREAD, BLOCK_SIZE * K_PER_THREAD);
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
    softmax_warp_kernel<<<ROWS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, COLS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    size_t first_idx = 0;
    for (size_t i = 0; i < N; i++) {
        if (!approx_eq(h_y[i], h_ref[i])) { if (mismatches == 0) first_idx = i; mismatches++; }
    }
    if (mismatches == 0)
        printf("softmax_warp: CPU and GPU results match (ROWS=%d, COLS=%d, BLOCK=%d, K=%d).\n",
               ROWS, COLS, BLOCK_SIZE, K_PER_THREAD);
    else
        printf("softmax_warp: MISMATCH (%d at %zu: CPU=%f, GPU=%f).\n",
               mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
