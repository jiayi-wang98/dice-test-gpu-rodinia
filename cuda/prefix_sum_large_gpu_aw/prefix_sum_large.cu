// prefix_sum_large.cu — N=65536 inclusive prefix scan via 3-kernel pipeline.
//
// K1 block_scan: each CTA does a Hillis-Steele inclusive scan on its 256
//                input elements, writes its local prefix to out[],
//                and writes its block-total to block_sums[blockIdx.x].
// K2 scan_block_sums: a single CTA scans the block_sums[] array
//                     (256 elements, fits in one CTA).
// K3 add_block_offsets: each CTA except 0 adds block_sums[blockIdx.x-1]
//                       to every element of its local prefix.
//
// Total: K1 runs grid=N/BLOCK_SIZE in parallel; K2 is single-CTA; K3 is
// grid=N/BLOCK_SIZE again. Matches the canonical large-array scan pattern
// (huffman's scanLargeArray and equivalents).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void block_scan_kernel(const int *in, int *out, int *block_sums, int N)
{
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    sdata[tid] = (gid < N) ? in[gid] : 0;
    __syncthreads();

    for (int d = 1; d < BLOCK_SIZE; d <<= 1) {
        __syncthreads();
        int v = sdata[tid];
        if (tid >= d) v += sdata[tid - d];
        __syncthreads();
        sdata[tid] = v;
    }
    __syncthreads();

    if (gid < N) out[gid] = sdata[tid];
    if (tid == BLOCK_SIZE - 1) block_sums[blockIdx.x] = sdata[tid];
}

__global__ void scan_block_sums_kernel(int *block_sums, int nblocks)
{
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    sdata[tid] = (tid < nblocks) ? block_sums[tid] : 0;
    __syncthreads();

    for (int d = 1; d < BLOCK_SIZE; d <<= 1) {
        __syncthreads();
        int v = sdata[tid];
        if (tid >= d) v += sdata[tid - d];
        __syncthreads();
        sdata[tid] = v;
    }
    __syncthreads();

    if (tid < nblocks) block_sums[tid] = sdata[tid];
}

__global__ void add_block_offsets_kernel(int *out, const int *block_sums, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Block 0 doesn't need an offset (its local inclusive scan is already
    // the global inclusive scan). Other blocks add the scanned sum of all
    // preceding blocks, which is block_sums[blockIdx.x - 1] after K2.
    if (blockIdx.x > 0 && gid < N) {
        out[gid] += block_sums[blockIdx.x - 1];
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
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nblocks > BLOCK_SIZE) {
        fprintf(stderr, "N too large: K2 expects nblocks (%d) <= BLOCK_SIZE (%d).\n",
                nblocks, BLOCK_SIZE);
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

    dim3 block(BLOCK_SIZE);
    dim3 grid_full(nblocks);
    dim3 grid_one(1);

    block_scan_kernel       <<<grid_full, block>>>(d_in, d_out, d_block_sums, N);
    scan_block_sums_kernel  <<<grid_one,  block>>>(d_block_sums, nblocks);
    add_block_offsets_kernel<<<grid_full, block>>>(d_out, d_block_sums, N);
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
        printf("prefix_sum_large: CPU and GPU results match (N=%d, last=%d).\n",
               N, h_ref[N-1]);
    } else {
        printf("prefix_sum_large: MISMATCH (N=%d, %d mismatches, first at i=%d: CPU=%d, GPU=%d).\n",
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
