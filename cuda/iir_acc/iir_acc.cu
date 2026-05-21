// iir_acc.cu — Multi-channel 1st-order IIR filter (DICE in-fabric
// feedback variant).
//
//   y[c, t] = a * x[c, t] + b * y[c, t-1],   y[c, -1] = 0
//
// DICE pattern: one CTA per channel, T threads per CTA dispatched in
// CTA-order at II=1.  An FMA-mode PE is configured with a feedback
// wire (switch-box bypass register, no new hardware) so that each
// dispatched thread reads the previous thread's output as its y_prev,
// computes y_new = a*x + b*y_prev, and feeds y_new back for the next
// thread.  The dice_loop_carry_fma intrinsic expresses this pattern.
//
// On a stock GPU this kernel would race (each thread's read and exch
// are separate instructions); we ship the GPU baseline as a separate
// source (iir.cu) that uses the production per-thread serial loop.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../dice_test/dice_atomics.h"

#ifndef T_PER_CTA
#define T_PER_CTA 512   // DICE per-CTA thread cap is 512
#endif

__global__ void iir_acc_kernel(const float *x, float *y,
                               float a, float b, int T)
{
    int c   = blockIdx.x;
    int tid = threadIdx.x;

    // Reset feedback register (slot 0) for this CTA's channel.
    if (tid == 0) ((float *)__dice_acc_slots)[0] = 0.0f;
    __syncthreads();

    // Each thread = one timestep.  The state-PE serialises dispatch in
    // tid order; the FMA+feedback computes the recurrence in-fabric.
    if (tid < T) {
        float x_t = x[(size_t)c * T + tid];
        float y_t;
        dice_loop_carry_fma(y_t, x_t, a, b);   // y_t = a*x_t + b*y_prev
        y[(size_t)c * T + tid] = y_t;
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
    if (C < 1 || T < 1 || T > T_PER_CTA) {
        fprintf(stderr, "T must be in [1, %d]\n", T_PER_CTA);
        return 1;
    }

    size_t N = (size_t)C * T;
    float *h_x   = (float *) malloc(N * sizeof(float));
    float *h_y   = (float *) malloc(N * sizeof(float));
    float *h_ref = (float *) malloc(N * sizeof(float));

    srand(9);
    for (size_t i = 0; i < N; i++) {
        h_x[i] = ((rand() & 0xFFFF) / 65535.0f) - 0.5f;
    }
    iir_cpu(h_x, h_ref, a, b, C, T);

    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // One CTA per channel; T threads per CTA (one timestep each).
    iir_acc_kernel<<<C, T_PER_CTA>>>(d_x, d_y, a, b, T);
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
        printf("iir_acc: CPU and GPU results match (C=%d, T=%d, last=%f).\n",
               C, T, h_ref[N-1]);
    } else {
        printf("iir_acc: MISMATCH (C=%d, T=%d, %d mismatches; first at %zu: CPU=%f, GPU=%f).\n",
               C, T, mismatches, first_idx, h_ref[first_idx], h_y[first_idx]);
    }

    cudaFree(d_x); cudaFree(d_y); free(h_x); free(h_y); free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
