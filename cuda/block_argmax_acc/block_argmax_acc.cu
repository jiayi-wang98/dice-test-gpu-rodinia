// block_argmax_acc.cu — DICE MAX-mode + MIN-mode state-PE variant.
//
// Two state-PE slots in two phases:
//   Slot 0 (MAX mode): per-CTA max value
//   Slot 1 (MIN mode): per-CTA argmax index (lowest tid whose v == max)
//
// Per-CTA partials at out[2*ctaid] = max, out[2*ctaid+1] = argmax index.
// Host folds for global argmax.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <climits>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_argmax_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Register-addressed slot init (workaround for DICETool symbol-store
    // bug; see workflow doc §6).
    if (tid < 2) {
        int *s = (int *)__dice_acc_slots;
        s[tid] = (tid == 0) ? INT_MIN : INT_MAX;
    }
    __syncthreads();

    int v = (gid < N) ? in[gid] : INT_MIN;

    // Pass 1: MAX-mode state-PE → slot 0.
    int prev_max = INT_MIN;
    dice_cta_acc_max(prev_max, v);
    __syncthreads();

    int the_max = ((int *)__dice_acc_slots)[0];

    // Pass 2: MIN-mode state-PE on tid where v == max → slot 1.
    int contribution = (v == the_max) ? tid : INT_MAX;
    int prev_idx = INT_MAX;
    dice_cta_acc_min(prev_idx, contribution);

    __syncthreads();
    if (tid == 0) {
        out[2 * blockIdx.x + 0] = ((int *)__dice_acc_slots)[0];
        out[2 * blockIdx.x + 1] = blockIdx.x * BLOCK_SIZE +
                                  ((int *)__dice_acc_slots)[1];
    }
}

void block_argmax_cpu(const int *in, int *max_out, int *idx_out, int N)
{
    int m = INT_MIN, mi = -1;
    for (int i = 0; i < N; i++) {
        if (in[i] > m) { m = in[i]; mi = i; }
    }
    *max_out = m; *idx_out = mi;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_in = (int *) malloc(N * sizeof(int));
    int ref_max = INT_MIN, ref_idx = -1;
    int *h_partials = (int *) malloc(2 * nblocks * sizeof(int));
    srand(9);
    for (int i = 0; i < N; i++) h_in[i] = (rand() & 0xFF) - 128;
    block_argmax_cpu(h_in, &ref_max, &ref_idx, N);

    int *d_in, *d_out;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out, 2 * nblocks * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 2 * nblocks * sizeof(int));

    block_argmax_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_partials, d_out, 2 * nblocks * sizeof(int), cudaMemcpyDeviceToHost);

    int gmax = INT_MIN, gidx = -1;
    for (int i = 0; i < nblocks; i++) {
        if (h_partials[2*i] > gmax ||
            (h_partials[2*i] == gmax && h_partials[2*i+1] < gidx)) {
            gmax = h_partials[2*i];
            gidx = h_partials[2*i+1];
        }
    }

    if (gmax == ref_max && gidx == ref_idx)
        printf("block_argmax_acc: CPU and GPU results match (N=%d, max=%d, idx=%d).\n",
               N, ref_max, ref_idx);
    else
        printf("block_argmax_acc: MISMATCH (N=%d, CPU=(%d,%d), GPU=(%d,%d)).\n",
               N, ref_max, ref_idx, gmax, gidx);

    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_partials);
    return (gmax == ref_max && gidx == ref_idx) ? 0 : 1;
}
