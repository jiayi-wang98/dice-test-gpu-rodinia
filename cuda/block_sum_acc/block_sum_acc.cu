// block_sum_acc.cu — block-sum reduction via DICE accumulator PE.
//
// One CTA, BLOCK_SIZE threads. Each thread strides through the input and
// feeds its partial sum into a CTA-local accumulator (dice_cta_acc_add).
// The last thread reads the final state and atomicAdd's to global.
//
// Replaces the SMEM tree-reduce in block_sum.cu with one PE-cycle per
// thread — no __syncthreads, no SMEM bank traffic, no warp shuffles.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_sum_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;

    // Phase 1: thread-strided accumulate into a per-thread partial.
    int local = 0;
    for (int i = tid; i < N; i += blockDim.x)
        local += in[i];

    // Phase 2: feed all partials into the accumulator PE.
    // On DICE the CGRA dispatches threads in deterministic CTA order, so the
    // last thread's X = total. On real GPUs atomic ordering is undefined, so
    // we don't rely on X; instead we use __syncthreads + a read of the slot.
    int X = 0;
    dice_cta_acc_add<0>(X, local);

    // Phase 3: barrier then publish total. On DICE the dispatch order makes
    // the explicit barrier a no-op; on real GPUs it's required to make
    // __dice_acc_slots[0] observable as the post-CTA total.
    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int)__dice_acc_slots[0]);
}

void block_sum_cpu(const int *in, int *out, int N)
{
    long long s = 0;
    for (int i = 0; i < N; i++) s += in[i];
    *out = (int)s;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) { fprintf(stderr, "N must be >= 1\n"); return 1; }

    int *h_in   = (int *) malloc(N * sizeof(int));
    int  h_out  = 0;
    int  h_ref  = 0;
    if (!h_in) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // Same seed as block_sum.cu so the two benchmarks compute the same sum.
    srand(9);
    for (int i = 0; i < N; i++)
        h_in[i] = (rand() & 0xFF) - 128;

    block_sum_cpu(h_in, &h_ref, N);

    int *d_in = NULL, *d_out = NULL;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out,     sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    dim3 grid(1);
    dim3 block(BLOCK_SIZE);
    block_sum_acc_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref) {
        printf("block_sum_acc: CPU and GPU results match (N=%d, sum=%d).\n",
               N, h_ref);
    } else {
        printf("block_sum_acc: MISMATCH (N=%d, CPU=%d, GPU=%d).\n",
               N, h_ref, h_out);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
