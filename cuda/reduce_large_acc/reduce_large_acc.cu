// reduce_large_acc.cu — N=65536 sum reduction across many CTAs, using
// the DICE state-PE intrinsic for the per-CTA reduce.
//
// Same multi-CTA structure as reduce_large.cu: each of nblocks=256 CTAs
// of 256 threads reduces 256 inputs to one partial sum, then thread 0
// publishes via atomicAdd to *out. The per-CTA reduction goes through
// dice_cta_acc_add (lowers to atom.shared.add on the magic per-CTA slot),
// replacing the log2(BLOCK_SIZE)-level SMEM tree.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void reduce_large_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // __shared__ is uninitialized per CUDA spec; thread 0 resets the slot.
    if (tid == 0) __dice_acc_slots[0] = 0;
    __syncthreads();

    int v = (gid < N) ? in[gid] : 0;

    int prefix = 0;
    dice_cta_acc_add(prefix, v);

    // The final accumulator state is the CTA's total. Thread 0 reads it
    // back from the slot and publishes via global atomic.
    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int)__dice_acc_slots[0]);
}

void reduce_cpu(const int *in, int *out, int N)
{
    long long s = 0;
    for (int i = 0; i < N; i++) s += in[i];
    *out = (int)s;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) { fprintf(stderr, "N must be >= 1\n"); return 1; }
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_in   = (int *) malloc(N * sizeof(int));
    int  h_out  = 0;
    int  h_ref  = 0;
    if (!h_in) { fprintf(stderr, "host alloc failed\n"); return 1; }

    srand(9);
    for (int i = 0; i < N; i++)
        h_in[i] = (rand() & 0xFF) - 128;

    reduce_cpu(h_in, &h_ref, N);

    int *d_in = NULL, *d_out = NULL;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out,     sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    dim3 grid(nblocks);
    dim3 block(BLOCK_SIZE);
    reduce_large_acc_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref) {
        printf("reduce_large_acc: CPU and GPU results match (N=%d, sum=%d).\n",
               N, h_ref);
    } else {
        printf("reduce_large_acc: MISMATCH (N=%d, CPU=%d, GPU=%d).\n",
               N, h_ref, h_out);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
