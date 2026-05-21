// l2norm_acc.cu — DICE state-PE variant of l2norm.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void l2norm_acc_kernel(const int *a, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid == 0) __dice_acc_slots[0] = 0;
    __syncthreads();

    int v = (gid < N) ? a[gid] : 0;
    int sq = v * v;
    int prefix = 0;
    dice_cta_acc_add(prefix, sq);

    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int)__dice_acc_slots[0]);
}

void l2_cpu(const int *a, int *out, int N)
{
    long long s = 0;
    for (int i = 0; i < N; i++) s += (long long)a[i] * a[i];
    *out = (int)s;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_a = (int *) malloc(N * sizeof(int));
    int  h_out = 0, h_ref = 0;
    srand(9);
    for (int i = 0; i < N; i++) h_a[i] = (rand() & 0xFF) - 128;
    l2_cpu(h_a, &h_ref, N);

    int *d_a, *d_out;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_out,    sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    l2norm_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref)
        printf("l2norm_acc: CPU and GPU results match (N=%d, sumsq=%d).\n", N, h_ref);
    else
        printf("l2norm_acc: MISMATCH (N=%d, CPU=%d, GPU=%d).\n", N, h_ref, h_out);

    cudaFree(d_a); cudaFree(d_out); free(h_a);
    return (h_out == h_ref) ? 0 : 1;
}
