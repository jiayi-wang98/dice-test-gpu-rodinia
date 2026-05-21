// dot_product_acc.cu — DICE state-PE variant of dot_product.
//
// Per-CTA: each thread computes a[i]*b[i] then feeds into a single
// state-PE accumulator slot via dice_cta_acc_add. Thread 0 publishes
// the per-CTA total via atomicAdd to a global counter.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void dot_product_acc_kernel(const int *a, const int *b,
                                       int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid == 0) __dice_acc_slots[0] = 0;
    __syncthreads();

    int prod = (gid < N) ? (a[gid] * b[gid]) : 0;
    int prefix = 0;
    dice_cta_acc_add(prefix, prod);

    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int)__dice_acc_slots[0]);
}

void dot_cpu(const int *a, const int *b, int *out, int N)
{
    long long s = 0;
    for (int i = 0; i < N; i++) s += (long long)a[i] * b[i];
    *out = (int)s;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_a = (int *) malloc(N * sizeof(int));
    int *h_b = (int *) malloc(N * sizeof(int));
    int  h_out = 0, h_ref = 0;
    srand(9);
    for (int i = 0; i < N; i++) h_a[i] = (rand() & 0xFF) - 128;
    for (int i = 0; i < N; i++) h_b[i] = (rand() & 0xFF) - 128;
    dot_cpu(h_a, h_b, &h_ref, N);

    int *d_a, *d_b, *d_out;
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_out,    sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    dot_product_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref)
        printf("dot_product_acc: CPU and GPU results match (N=%d, dot=%d).\n", N, h_ref);
    else
        printf("dot_product_acc: MISMATCH (N=%d, CPU=%d, GPU=%d).\n", N, h_ref, h_out);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b);
    return (h_out == h_ref) ? 0 : 1;
}
