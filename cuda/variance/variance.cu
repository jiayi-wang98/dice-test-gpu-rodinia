// variance.cu — one-pass mean+variance: emits Σx and Σx² per array.
//
// Per-CTA: two parallel SMEM-tree reductions, one for sum and one for
// sum-of-squares. Thread 0 atomicAdd's both to global counters
// out[0] = Σx, out[1] = Σx². The host computes μ = Σx/N and
// σ² = Σx²/N - μ².

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void variance_kernel(const int *a, int *out, int N)
{
    __shared__ int s_sum[BLOCK_SIZE];
    __shared__ int s_sq [BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    int v = (gid < N) ? a[gid] : 0;
    s_sum[tid] = v;
    s_sq [tid] = v * v;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq [tid] += s_sq [tid + s];
        }
        __syncthreads();
    }

    // Per-CTA partials (sum, sumsq) at out[2*ctaid + 0/1]. Host sums.
    // Avoids two consecutive atom.global.add ops in the same DBB, which
    // overflow DICE's LDST per-port access queue.
    if (tid == 0) {
        out[2 * blockIdx.x + 0] = s_sum[0];
        out[2 * blockIdx.x + 1] = s_sq [0];
    }
}

void var_cpu(const int *a, int *out_sum, int *out_sq, int N)
{
    long long ss = 0, sq = 0;
    for (int i = 0; i < N; i++) { ss += a[i]; sq += (long long)a[i]*a[i]; }
    *out_sum = (int)ss; *out_sq = (int)sq;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_a = (int *) malloc(N * sizeof(int));
    int h_out[2] = {0, 0}, h_ref_sum = 0, h_ref_sq = 0;
    srand(9);
    for (int i = 0; i < N; i++) h_a[i] = (rand() & 0xFF) - 128;
    var_cpu(h_a, &h_ref_sum, &h_ref_sq, N);

    int *d_a, *d_out;
    int *h_partials = (int *) malloc(2 * nblocks * sizeof(int));
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_out, 2 * nblocks * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 2 * nblocks * sizeof(int));

    variance_kernel<<<nblocks, BLOCK_SIZE>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_partials, d_out, 2 * nblocks * sizeof(int),
               cudaMemcpyDeviceToHost);
    h_out[0] = h_out[1] = 0;
    for (int i = 0; i < nblocks; i++) {
        h_out[0] += h_partials[2*i + 0];
        h_out[1] += h_partials[2*i + 1];
    }

    if (h_out[0] == h_ref_sum && h_out[1] == h_ref_sq)
        printf("variance: CPU and GPU results match (N=%d, sum=%d, sumsq=%d).\n",
               N, h_ref_sum, h_ref_sq);
    else
        printf("variance: MISMATCH (N=%d, CPU=(%d,%d), GPU=(%d,%d)).\n",
               N, h_ref_sum, h_ref_sq, h_out[0], h_out[1]);

    cudaFree(d_a); cudaFree(d_out); free(h_a); free(h_partials);
    return (h_out[0] == h_ref_sum && h_out[1] == h_ref_sq) ? 0 : 1;
}
