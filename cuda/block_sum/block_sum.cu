// block_sum.cu — single-CTA reduction microbenchmark.
//
// Sums N integers using a single CTA of BLOCK_SIZE threads, via the
// classic SMEM tree-reduce pattern (no warp shuffles — simpler for
// gpgpu-sim correctness). One thread then atomicAdd's to global.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_sum_kernel(const int *in, int *out, int N)
{
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Phase 1: thread-strided accumulate.
    int local = 0;
    for (int i = tid; i < N; i += blockDim.x)
        local += in[i];
    sdata[tid] = local;
    __syncthreads();

    // Phase 2: SMEM tree reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Phase 3: thread 0 publishes to global.
    if (tid == 0)
        atomicAdd(out, sdata[0]);
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

    // Deterministic init: same seed -> same input every run.
    srand(9);
    for (int i = 0; i < N; i++)
        h_in[i] = (rand() & 0xFF) - 128;     // signed range [-128, 127]

    block_sum_cpu(h_in, &h_ref, N);

    int *d_in = NULL, *d_out = NULL;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out,     sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    dim3 grid(1);
    dim3 block(BLOCK_SIZE);
    block_sum_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref) {
        printf("block_sum: CPU and GPU results match (N=%d, sum=%d).\n",
               N, h_ref);
    } else {
        printf("block_sum: MISMATCH (N=%d, CPU=%d, GPU=%d).\n",
               N, h_ref, h_out);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
