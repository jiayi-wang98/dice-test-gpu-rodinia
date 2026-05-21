// block_max_acc.cu — DICE MAX-mode state-PE variant of block_max.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <climits>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_max_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Initialise slot 0 to INT_MIN (so first atomicMax produces a
    // meaningful comparison). Register-addressed store to dodge
    // DICETool's symbol-addressed-store bug.
    if (tid < 1) ((int *)__dice_acc_slots)[tid] = INT_MIN;
    __syncthreads();

    int v = (gid < N) ? in[gid] : INT_MIN;
    int prev = INT_MIN;
    dice_cta_acc_max(prev, v);   // → MAX-mode state-PE, slot 0

    __syncthreads();
    if (tid == 0)
        atomicMax(out, ((int *)__dice_acc_slots)[0]);
}

void block_max_cpu(const int *in, int *out, int N)
{
    int m = INT_MIN;
    for (int i = 0; i < N; i++) if (in[i] > m) m = in[i];
    *out = m;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_in = (int *) malloc(N * sizeof(int));
    int h_out = INT_MIN, h_ref = INT_MIN;
    srand(9);
    for (int i = 0; i < N; i++) h_in[i] = (rand() & 0xFF) - 128;
    block_max_cpu(h_in, &h_ref, N);

    int *d_in, *d_out;
    int init_min = INT_MIN;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out,    sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &init_min, sizeof(int), cudaMemcpyHostToDevice);

    block_max_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref)
        printf("block_max_acc: CPU and GPU results match (N=%d, max=%d).\n", N, h_ref);
    else
        printf("block_max_acc: MISMATCH (N=%d, CPU=%d, GPU=%d).\n", N, h_ref, h_out);

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
