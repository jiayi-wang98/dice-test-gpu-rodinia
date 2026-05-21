// block_argmax.cu — per-CTA (max, argmax) reduction.
//
// Two-pass within a single CTA:
//   Pass 1: SMEM tree max → max value
//   Pass 2: each thread whose value equals max competes via atomicMin
//           on its tid → first (lowest) index with the max
//
// Per-CTA partials at out[2*ctaid] = max, out[2*ctaid+1] = argmax.
// Host folds across CTAs for the global argmax.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <climits>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_argmax_kernel(const int *in, int *out, int N)
{
    __shared__ int sdata[BLOCK_SIZE];
    __shared__ int s_max;
    __shared__ int s_argmax;
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    int v = (gid < N) ? in[gid] : INT_MIN;
    sdata[tid] = v;
    if (tid == 0) { s_max = INT_MIN; s_argmax = INT_MAX; }
    __syncthreads();

    // Pass 1: SMEM tree max.
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) s_max = sdata[0];
    __syncthreads();

    // Pass 2: argmax via atomicMin on (tid where v == max).
    if (v == s_max) atomicMin(&s_argmax, tid);
    __syncthreads();

    if (tid == 0) {
        out[2 * blockIdx.x + 0] = s_max;
        out[2 * blockIdx.x + 1] = blockIdx.x * BLOCK_SIZE + s_argmax;
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

    block_argmax_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
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
        printf("block_argmax: CPU and GPU results match (N=%d, max=%d, idx=%d).\n",
               N, ref_max, ref_idx);
    else
        printf("block_argmax: MISMATCH (N=%d, CPU=(%d,%d), GPU=(%d,%d)).\n",
               N, ref_max, ref_idx, gmax, gidx);

    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_partials);
    return (gmax == ref_max && gidx == ref_idx) ? 0 : 1;
}
