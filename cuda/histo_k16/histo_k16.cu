// histo_k16.cu — K=16 fixed-bin histogram with per-thread atomicAdd on
// SMEM bins.
//
// Each thread takes its input value, hashes it via ((val + 128) >> 4) & 0xF
// into a bin index in [0, 15], then atomicAdd's 1 to the per-CTA bin
// counter. After the within-CTA accumulation, thread k publishes bin k to
// global via atomicAdd. This is a small-K analogue of Parboil histo
// (where K=256 is runtime-indexed); we use K=16 specifically because it
// fits within state-PE's compile-time slot budget on DICE.
//
// Hash output is per-thread data-dependent. The GPU baseline uses
// atomic.shared.add on a data-dependent index; the DICE+state-PE variant
// must enumerate all 16 slots per thread (each thread runs 16 conditional
// state-PE ops, 15 add 0 and 1 adds 1 — a known overhead of the
// compile-time-slot restriction, documented in the paper).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define NBINS 16

__global__ void histo_kernel(const int *in, int *out, int N)
{
    __shared__ int s_bins[NBINS];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid < NBINS) s_bins[tid] = 0;
    __syncthreads();

    if (gid < N) {
        int v = in[gid];
        int bin = ((v + 128) >> 4) & 0xF;
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();

    if (tid < NBINS) {
        atomicAdd(&out[tid], s_bins[tid]);
    }
}

void histo_cpu(const int *in, int *out, int N)
{
    for (int k = 0; k < NBINS; k++) out[k] = 0;
    for (int i = 0; i < N; i++) {
        int bin = ((in[i] + 128) >> 4) & 0xF;
        out[bin]++;
    }
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) return 1;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *h_in = (int *) malloc(N * sizeof(int));
    int h_out[NBINS] = {0}, h_ref[NBINS] = {0};
    srand(9);
    for (int i = 0; i < N; i++) h_in[i] = (rand() & 0xFF) - 128;
    histo_cpu(h_in, h_ref, N);

    int *d_in, *d_out;
    cudaMalloc((void **)&d_in,  N * sizeof(int));
    cudaMalloc((void **)&d_out, NBINS * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, NBINS * sizeof(int));

    histo_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NBINS * sizeof(int), cudaMemcpyDeviceToHost);

    int ok = 1;
    for (int k = 0; k < NBINS; k++) if (h_out[k] != h_ref[k]) { ok = 0; break; }
    if (ok) {
        printf("histo_k16: CPU and GPU results match (N=%d, K=%d, sum=%d).\n",
               N, NBINS, N);
    } else {
        printf("histo_k16: MISMATCH (N=%d):\n  CPU=", N);
        for (int k = 0; k < NBINS; k++) printf(" %d", h_ref[k]);
        printf("\n  GPU=");
        for (int k = 0; k < NBINS; k++) printf(" %d", h_out[k]);
        printf("\n");
    }

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return ok ? 0 : 1;
}
