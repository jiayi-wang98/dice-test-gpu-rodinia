// variance_acc.cu — DICE state-PE variant of variance.
//
// Two simultaneous state-PE slots in one DBB: slot 0 accumulates Σx,
// slot 1 accumulates Σx². The __COUNTER__-based dice_cta_acc_add macro
// hands out distinct slot indices automatically.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dice_test/dice_atomics.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void variance_acc_kernel(const int *a, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Reset slots using tid as the index — forces register-addressed
    // SMEM store, which the DICETool emits correctly. Symbol-addressed
    // stores (e.g., `__dice_acc_slots[0] = 0`) currently mis-render in
    // the PPTX (address and value get swapped); known DICETool bug.
    if (tid < 2) __dice_acc_slots[tid] = 0;
    __syncthreads();

    int v = (gid < N) ? a[gid] : 0;
    int sq = v * v;

    // Two state-PE accumulators in the same DBB.
    int p0 = 0, p1 = 0;
    dice_cta_acc_add(p0, v);   // → slot 0
    dice_cta_acc_add(p1, sq);  // → slot 1

    __syncthreads();
    // Per-CTA partials to avoid two consecutive atom.global.add in the
    // same DBB (DICE LDST per-port queue cap).
    if (tid == 0) {
        out[2 * blockIdx.x + 0] = (int)__dice_acc_slots[0];
        out[2 * blockIdx.x + 1] = (int)__dice_acc_slots[1];
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

    variance_acc_kernel<<<nblocks, BLOCK_SIZE>>>(d_a, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_partials, d_out, 2 * nblocks * sizeof(int),
               cudaMemcpyDeviceToHost);
    h_out[0] = h_out[1] = 0;
    for (int i = 0; i < nblocks; i++) {
        h_out[0] += h_partials[2*i + 0];
        h_out[1] += h_partials[2*i + 1];
    }

    if (h_out[0] == h_ref_sum && h_out[1] == h_ref_sq)
        printf("variance_acc: CPU and GPU results match (N=%d, sum=%d, sumsq=%d).\n",
               N, h_ref_sum, h_ref_sq);
    else
        printf("variance_acc: MISMATCH (N=%d, CPU=(%d,%d), GPU=(%d,%d)).\n",
               N, h_ref_sum, h_ref_sq, h_out[0], h_out[1]);

    cudaFree(d_a); cudaFree(d_out); free(h_a); free(h_partials);
    return (h_out[0] == h_ref_sum && h_out[1] == h_ref_sq) ? 0 : 1;
}
