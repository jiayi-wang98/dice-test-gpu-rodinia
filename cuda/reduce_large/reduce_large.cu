// reduce_large.cu — N=65536 sum reduction across many CTAs.
//
// Multi-CTA reduction baseline using the classic SMEM-tree pattern per
// CTA, followed by an atomicAdd to a single global counter. Each of
// nblocks=256 CTAs of 256 threads reduces 256 inputs to one partial sum
// via a log2(256)=8-level Hillis-Steele-style SMEM tree, then thread 0
// publishes via atomicAdd to *out.
//
// The state-PE variant (reduce_large_acc) replaces the per-CTA SMEM tree
// with a single dice_cta_acc_add — see reduce_large_acc.cu.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void reduce_large_kernel(const int *in, int *out, int N)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    sdata[tid] = (gid < N) ? in[gid] : 0;
    __syncthreads();

    // SMEM tree reduction.
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, sdata[0]);
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
    reduce_large_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref) {
        printf("reduce_large: CPU and GPU results match (N=%d, sum=%d).\n",
               N, h_ref);
    } else {
        printf("reduce_large: MISMATCH (N=%d, CPU=%d, GPU=%d).\n",
               N, h_ref, h_out);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
