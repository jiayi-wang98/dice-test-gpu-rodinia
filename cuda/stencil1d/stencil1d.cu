// stencil1d: 3-point 1-D Jacobi over an int array.
// Single-iteration: out[i] = (in[i-1] + in[i] + in[i+1]) for 0 < i < N-1
// Boundary: out[0] = in[0] + in[1]; out[N-1] = in[N-2] + in[N-1].
//
// Per-block tile of BLOCK_SIZE elements; halo-1 each side.
// Mirrors the SMEM round-trip pattern from pathfinder so the .meta/.pptx
// shape is identical to a real Rodinia kernel.
//
// The kernel is the SMEM-baseline version. The RF-unified version is
// hand-written in stencil1d.1.sm_52.rfu.{meta,pptx}; it is not produced
// by nvcc because it depends on the cross-thread RF read primitive.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define BLOCK_SIZE 256
#define HALO 1
#define DEVICE 0
#define M_SEED 9

__global__ void stencil1d_kernel(const int *in, int *out, int N)
{
    __shared__ int tile[BLOCK_SIZE];

    int tx = threadIdx.x;
    int gx = blockIdx.x * BLOCK_SIZE + tx;

    // STS: each thread loads its own element; out-of-range stays 0.
    int v = (gx < N) ? in[gx] : 0;
    tile[tx] = v;
    __syncthreads();

    if (gx >= N) return;

    int center = tile[tx];
    int left   = (tx > 0)            ? tile[tx-1]
               : ((gx > 0)           ? in[gx-1]   : center);
    int right  = (tx < BLOCK_SIZE-1) ? tile[tx+1]
               : ((gx < N-1)         ? in[gx+1]   : center);

    out[gx] = left + center + right;
}

void stencil1d_cpu(const int *in, int *out, int N)
{
    for (int i = 0; i < N; i++) {
        int c = in[i];
        int l = (i > 0)   ? in[i-1] : c;
        int r = (i < N-1) ? in[i+1] : c;
        out[i] = l + c + r;
    }
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N < 2) { fprintf(stderr, "N must be >= 2\n"); return 1; }

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    size_t bytes = (size_t)N * sizeof(int);
    int *h_in   = (int *)malloc(bytes);
    int *h_out  = (int *)malloc(bytes);
    int *h_ref  = (int *)malloc(bytes);

    srand(M_SEED);
    for (int i = 0; i < N; i++) h_in[i] = rand() % 10;

    int *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil1d_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    stencil1d_cpu(h_in, h_ref, N);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i]) {
            if (mismatches < 8)
                printf("Mismatch at %d: GPU=%d CPU=%d\n", i, h_out[i], h_ref[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("stencil1d: CPU and GPU results match (N=%d).\n", N);
    } else {
        printf("stencil1d: %d mismatches (N=%d). FAIL.\n", mismatches, N);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);

    return mismatches == 0 ? 0 : 1;
}
