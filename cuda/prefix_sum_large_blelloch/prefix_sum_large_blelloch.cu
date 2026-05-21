// prefix_sum_large_blelloch.cu — N=65536 inclusive prefix scan using a
// work-optimal Blelloch up-sweep/down-sweep tree. Apples-to-apples with
// prefix_sum_large (Hillis-Steele): same N, same 3-kernel pipeline
// (block-scan, scan-block-sums, add-offsets), same input distribution.
//
// K1 (block_scan_blelloch): 128 threads per block scan 256 elements with
//   Blelloch (log2(256)=8 up-sweep + 8 down-sweep levels). Inclusive
//   output produced by inclusive = exclusive + input.
// K2 (scan_block_sums_blelloch): single CTA scans the 256 block totals
//   with the same Blelloch kernel (256 elements, 128 threads).
// K3 (add_block_offsets): unchanged from prefix_sum_large.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif
#define ELEMS_PER_BLOCK (BLOCK_THREADS * 2)

__global__ void block_scan_blelloch_kernel(const int *in, int *out, int *block_sums, int N)
{
    __shared__ int sdata[ELEMS_PER_BLOCK];
    __shared__ int s_block_total;
    int tid = threadIdx.x;
    int block_base = blockIdx.x * ELEMS_PER_BLOCK;

    int a = block_base + tid;
    int b = block_base + tid + BLOCK_THREADS;
    int ina = (a < N) ? in[a] : 0;
    int inb = (b < N) ? in[b] : 0;
    sdata[tid] = ina;
    sdata[tid + BLOCK_THREADS] = inb;
    __syncthreads();

    // Up-sweep: all threads compute safe ai/bi via predicate-on-index,
    // unconditionally read, conditionally write back. This matches the
    // predicated-execution pattern DICE's CGRA dispatch handles cleanly.
    int offset = 1;
    for (int d = ELEMS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        bool active = (tid < d);
        int ai = active ? (offset * (2 * tid + 1) - 1) : 0;
        int bi = active ? (offset * (2 * tid + 2) - 1) : 0;
        int va = sdata[ai];
        int vb = sdata[bi];
        int newval = vb + va;
        __syncthreads();
        if (active) sdata[bi] = newval;
        __syncthreads();
        offset <<= 1;
    }

    // Save block total in SMEM (every thread reads it back later if needed),
    // then clear last element. Single-thread predicate replaced by an
    // explicit SMEM communication slot.
    if (tid == 0) {
        s_block_total = sdata[ELEMS_PER_BLOCK - 1];
        sdata[ELEMS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    // Down-sweep: same predicated-index, unconditional-read,
    // conditional-write pattern.
    for (int d = 1; d < ELEMS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        bool active = (tid < d);
        int ai = active ? (offset * (2 * tid + 1) - 1) : 0;
        int bi = active ? (offset * (2 * tid + 2) - 1) : 0;
        int t  = sdata[ai];
        int tb = sdata[bi];
        __syncthreads();
        if (active) {
            sdata[ai] = tb;
            sdata[bi] = tb + t;
        }
        __syncthreads();
    }

    // Convert exclusive → inclusive by adding the input element.
    if (a < N) out[a] = sdata[tid] + ina;
    if (b < N) out[b] = sdata[tid + BLOCK_THREADS] + inb;

    // Write block total — all threads write the same value to the same
    // address (idempotent). Avoids a single-thread predicated store, which
    // the DICE simulator's predicated cvta path doesn't handle cleanly
    // for this kernel.
    block_sums[blockIdx.x] = s_block_total;
}

__global__ void scan_block_sums_blelloch_kernel(int *block_sums, int nblocks)
{
    __shared__ int sdata[ELEMS_PER_BLOCK];
    int tid = threadIdx.x;

    int a = tid;
    int b = tid + BLOCK_THREADS;
    int ina = (a < nblocks) ? block_sums[a] : 0;
    int inb = (b < nblocks) ? block_sums[b] : 0;
    sdata[tid] = ina;
    sdata[tid + BLOCK_THREADS] = inb;
    __syncthreads();

    int offset = 1;
    for (int d = ELEMS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        bool active = (tid < d);
        int ai = active ? (offset * (2 * tid + 1) - 1) : 0;
        int bi = active ? (offset * (2 * tid + 2) - 1) : 0;
        int va = sdata[ai];
        int vb = sdata[bi];
        int newval = vb + va;
        __syncthreads();
        if (active) sdata[bi] = newval;
        __syncthreads();
        offset <<= 1;
    }
    if (tid == 0) sdata[ELEMS_PER_BLOCK - 1] = 0;
    __syncthreads();
    for (int d = 1; d < ELEMS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        bool active = (tid < d);
        int ai = active ? (offset * (2 * tid + 1) - 1) : 0;
        int bi = active ? (offset * (2 * tid + 2) - 1) : 0;
        int t  = sdata[ai];
        int tb = sdata[bi];
        __syncthreads();
        if (active) {
            sdata[ai] = tb;
            sdata[bi] = tb + t;
        }
        __syncthreads();
    }

    if (a < nblocks) block_sums[a] = sdata[tid] + ina;
    if (b < nblocks) block_sums[b] = sdata[tid + BLOCK_THREADS] + inb;
}

__global__ void add_block_offsets_kernel(int *out, const int *block_sums, int N)
{
    int tid = threadIdx.x;
    int block_base = blockIdx.x * ELEMS_PER_BLOCK;
    int a = block_base + tid;
    int b = block_base + tid + BLOCK_THREADS;

    if (blockIdx.x > 0) {
        int off = block_sums[blockIdx.x - 1];
        if (a < N) out[a] += off;
        if (b < N) out[b] += off;
    }
}

void prefix_sum_cpu(const int *in, int *out, int N)
{
    long long s = 0;
    for (int i = 0; i < N; i++) {
        s += in[i];
        out[i] = (int)s;
    }
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 65536;
    if (N < 1) { fprintf(stderr, "N must be >= 1\n"); return 1; }
    int nblocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    if (nblocks > ELEMS_PER_BLOCK) {
        fprintf(stderr, "N too large: K2 expects nblocks (%d) <= ELEMS_PER_BLOCK (%d).\n",
                nblocks, ELEMS_PER_BLOCK);
        return 1;
    }

    int *h_in   = (int *) malloc(N * sizeof(int));
    int *h_out  = (int *) malloc(N * sizeof(int));
    int *h_ref  = (int *) malloc(N * sizeof(int));
    if (!h_in || !h_out || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    srand(9);
    for (int i = 0; i < N; i++)
        h_in[i] = (rand() & 0xFF) - 128;

    prefix_sum_cpu(h_in, h_ref, N);

    int *d_in = NULL, *d_out = NULL, *d_block_sums = NULL;
    cudaMalloc((void **)&d_in,         N * sizeof(int));
    cudaMalloc((void **)&d_out,        N * sizeof(int));
    cudaMalloc((void **)&d_block_sums, nblocks * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));
    cudaMemset(d_block_sums, 0, nblocks * sizeof(int));

    dim3 block(BLOCK_THREADS);
    dim3 grid_full(nblocks);
    dim3 grid_one(1);

    block_scan_blelloch_kernel       <<<grid_full, block>>>(d_in, d_out, d_block_sums, N);
    scan_block_sums_blelloch_kernel  <<<grid_one,  block>>>(d_block_sums, nblocks);
    add_block_offsets_kernel         <<<grid_full, block>>>(d_out, d_block_sums, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    int first_mismatch_idx = -1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i]) {
            if (mismatches == 0) first_mismatch_idx = i;
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("prefix_sum_large_blelloch: CPU and GPU results match (N=%d, last=%d).\n",
               N, h_ref[N-1]);
    } else {
        printf("prefix_sum_large_blelloch: MISMATCH (N=%d, %d mismatches, first at i=%d: CPU=%d, GPU=%d).\n",
               N, mismatches, first_mismatch_idx,
               h_ref[first_mismatch_idx], h_out[first_mismatch_idx]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    free(h_in);
    free(h_out);
    free(h_ref);
    return (mismatches == 0) ? 0 : 1;
}
