// block_max.cu — c = max(a[i]) for N integers.
//
// Multi-CTA block-max baseline: each CTA reduces 256 inputs to a per-CTA
// max via SMEM tree, then thread 0 publishes via atomicMax to a single
// global counter.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <climits>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_max_kernel(const int *in, int *out, int N)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    sdata[tid] = (gid < N) ? in[gid] : INT_MIN;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int b = sdata[tid + s];
            int a = sdata[tid];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) atomicMax(out, sdata[0]);
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

    block_max_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out == h_ref)
        printf("block_max: CPU and GPU results match (N=%d, max=%d).\n", N, h_ref);
    else
        printf("block_max: MISMATCH (N=%d, CPU=%d, GPU=%d).\n", N, h_ref, h_out);

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return (h_out == h_ref) ? 0 : 1;
}
