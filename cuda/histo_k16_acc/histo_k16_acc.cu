// histo_k16_acc.cu — DICE state-PE variant of histo_k16.
//
// state-PE requires compile-time-known slot IDs; the hash output is
// per-thread runtime data. We unroll 16-way: each thread runs 16
// dice_cta_acc_add calls (one per slot), passing the value 1 only when
// its bin matches the slot index and 0 otherwise. 15 of the 16 ops add 0
// per thread. This is the cost of fitting a data-indexed atomic into a
// compile-time-slot mechanism; the paper discusses this in §V.D as a
// known limitation of the current state-PE microarch (a "dynamic slot
// index" mode would eliminate the 16× overhead but is left to future
// work).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define NBINS 16

template <int K>
__device__ __forceinline__ void enumerate_acc(int bin)
{
    int contribution = (bin == K) ? 1 : 0;
    dice_cta_acc_add_slot<K>(contribution, contribution);
}

__global__ void histo_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid < NBINS) __dice_acc_slots[tid] = 0;
    __syncthreads();

    int v = (gid < N) ? in[gid] : 0;
    int bin = (gid < N) ? (((v + 128) >> 4) & 0xF) : -1;

    // 16-way enumeration: every thread does 16 state-PE ops, only the
    // one matching its bin adds 1.
    enumerate_acc< 0>(bin); enumerate_acc< 1>(bin);
    enumerate_acc< 2>(bin); enumerate_acc< 3>(bin);
    enumerate_acc< 4>(bin); enumerate_acc< 5>(bin);
    enumerate_acc< 6>(bin); enumerate_acc< 7>(bin);
    enumerate_acc< 8>(bin); enumerate_acc< 9>(bin);
    enumerate_acc<10>(bin); enumerate_acc<11>(bin);
    enumerate_acc<12>(bin); enumerate_acc<13>(bin);
    enumerate_acc<14>(bin); enumerate_acc<15>(bin);

    __syncthreads();
    if (tid < NBINS) {
        atomicAdd(&out[tid], (int)__dice_acc_slots[tid]);
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

    histo_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NBINS * sizeof(int), cudaMemcpyDeviceToHost);

    int ok = 1;
    for (int k = 0; k < NBINS; k++) if (h_out[k] != h_ref[k]) { ok = 0; break; }
    if (ok) {
        printf("histo_k16_acc: CPU and GPU results match (N=%d, K=%d).\n", N, NBINS);
    } else {
        printf("histo_k16_acc: MISMATCH (N=%d):\n  CPU=", N);
        for (int k = 0; k < NBINS; k++) printf(" %d", h_ref[k]);
        printf("\n  GPU=");
        for (int k = 0; k < NBINS; k++) printf(" %d", h_out[k]);
        printf("\n");
    }

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return ok ? 0 : 1;
}
