// iir.cu — Multi-channel 1st-order IIR filter (GPU SIMT baseline).
//
//   y[c, t] = a * x[c, t] + b * y[c, t-1],   y[c, -1] = 0
//
// GPU pattern: one thread per channel, serial T-loop in the thread.
// The per-channel recurrence is loop-carried (y[t] depends on y[t-1])
// so it cannot be SIMT-parallelised within a channel; parallelism is
// across channels only.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void iir_kernel(const float *x, float *y,
                           float a, float b, int C, int T)
{
    int c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (c >= C) return;

    float y_prev = 0.0f;
    const float *xp = x + (size_t)c * T;
    float       *yp = y + (size_t)c * T;
    for (int t = 0; t < T; t++) {
        float yt = a * xp[t] + b * y_prev;
        yp[t] = yt;
        y_prev = yt;
    }
}

void iir_cpu(const float *x, float *y, float a, float b, int C, int T)
{
    for (int c = 0; c < C; c++) {
        float y_prev = 0.0f;
        for (int t = 0; t < T; t++) {
            float yt = a * x[(size_t)c * T + t] + b * y_prev;
            y[(size_t)c * T + t] = yt;
            y_prev = yt;
        }
    }
}

static int approx_eq(float a, float b)
{
    float diff = fabsf(a - b);
    float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return diff <= 1e-3f * scale;
}

int main(int argc, char **argv)
{
    int C = (argc > 1) ? atoi(argv[1]) : 256;
    int T = (argc > 2) ? atoi(argv[2]) : 1024;
    float a = 0.5f, b = 0.5f;
    if (C < 1 || T < 1) return 1;

    size_t N = (size_t)C * T;
    float *h_x   = (float *) malloc(N * sizeof(float));
    float *h_y   = (float *) malloc(N * sizeof(float));
    float *h_ref = (float *) malloc(N * sizeof(float));

    srand(9);
    for (size_t i = 0; i < N; i++) {
        h_x[i] = ((rand() & 0xFFFF) / 65535.0f) - 0.5f;  // uniform in [-0.5, 0.5]
    }
    iir_cpu(h_x, h_ref, a, b, C, T);

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int nblocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
    iir_kernel<<<nblocks, BLOCK_SIZE>>>(d_x, d_y, a, b, C, T);
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
        printf("iir: CPU and GPU results match (C=%d, T=%d, last=%f).\n",
               C, T, h_ref[N-1]);
    } else {
        printf("iir: MISMATCH (C=%d, T=%d, %d mismatches; first at %zu: CPU=%f, GPU=%f).\n",
               C, T, mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);
    }

    cudaFree(d_x); cudaFree(d_y); free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
