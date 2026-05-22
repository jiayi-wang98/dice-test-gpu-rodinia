// softmax.cu — row-wise numerically-stable softmax over a batch of
// rows (GPU SIMT baseline, production SMEM-tree pattern).
//
// One CTA per row, BLOCK_SIZE threads per CTA.  Three passes:
//   Pass 1: tree reduction over the row to find the per-row max m.
//   Pass 2: tree reduction over the row of exp(x - m) to find sum s.
//   Pass 3: per-thread divide y[i] = exp(x[i] - m) / s.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

__global__ void softmax_kernel(const float *x, float *y, int COLS)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float xi = x[(size_t)row * COLS + tid];

    // Pass 1: SMEM tree max reduction
    smem[tid] = xi;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float m = smem[0];
    __syncthreads();

    // Pass 2: SMEM tree sum reduction of exp(xi - m)
    float e = __expf(xi - m);
    smem[tid] = e;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float s_row = smem[0];

    // Pass 3: divide
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

    softmax_kernel<<<ROWS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>
        (d_x, d_y, COLS);
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
        printf("softmax: CPU and GPU results match (ROWS=%d, COLS=%d).\n",
               ROWS, COLS);
    } else {
        printf("softmax: MISMATCH (ROWS=%d, COLS=%d, %d mismatches; first at %zu: CPU=%f, GPU=%f).\n",
               ROWS, COLS, mismatches, first_idx,
               h_ref[first_idx], h_y[first_idx]);
    }

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
