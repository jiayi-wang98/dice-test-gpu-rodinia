// stencil2d: 5-point 2-D Jacobi over an int field.
// out[i,j] = in[i-1,j] + in[i+1,j] + in[i,j-1] + in[i,j+1] + in[i,j]
// Boundary: replicate (clamp).
//
// Block layout: BLOCK_X x BLOCK_Y = 16 x 16 = 256 threads.
// SMEM tile is plain (BLOCK_X x BLOCK_Y) — no halo padding. Out-of-block
// neighbours come from gmem; this matches the simplest hotspot variant
// and keeps the SMEM footprint a power-of-two row stride for the
// bank-conflict discussion in the .pptx.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define DEVICE 0
#define M_SEED 9

__global__ void stencil2d_kernel(const int *in, int *out, int W, int H)
{
    __shared__ int tile[BLOCK_Y][BLOCK_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * BLOCK_X + tx;
    int gy = blockIdx.y * BLOCK_Y + ty;

    int v = (gx < W && gy < H) ? in[gy * W + gx] : 0;
    tile[ty][tx] = v;
    __syncthreads();

    if (gx >= W || gy >= H) return;

    // Within-block neighbours from SMEM; cross-block neighbours from gmem.
    int center = tile[ty][tx];
    int up    = (ty > 0)         ? tile[ty-1][tx] : ((gy > 0)   ? in[(gy-1) * W + gx] : center);
    int down  = (ty < BLOCK_Y-1) ? tile[ty+1][tx] : ((gy < H-1) ? in[(gy+1) * W + gx] : center);
    int left  = (tx > 0)         ? tile[ty][tx-1] : ((gx > 0)   ? in[gy * W + (gx-1)] : center);
    int right = (tx < BLOCK_X-1) ? tile[ty][tx+1] : ((gx < W-1) ? in[gy * W + (gx+1)] : center);

    out[gy * W + gx] = up + down + left + right + center;
}

void stencil2d_cpu(const int *in, int *out, int W, int H)
{
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int c = in[j * W + i];
            int u = (j > 0)   ? in[(j-1) * W + i] : c;
            int d = (j < H-1) ? in[(j+1) * W + i] : c;
            int l = (i > 0)   ? in[j * W + (i-1)] : c;
            int r = (i < W-1) ? in[j * W + (i+1)] : c;
            out[j * W + i] = u + d + l + r + c;
        }
    }
}

int main(int argc, char **argv)
{
    int W = (argc > 1) ? atoi(argv[1]) : 64;
    int H = (argc > 2) ? atoi(argv[2]) : 64;

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    size_t bytes = (size_t)W * H * sizeof(int);
    int *h_in   = (int *)malloc(bytes);
    int *h_out  = (int *)malloc(bytes);
    int *h_ref  = (int *)malloc(bytes);

    srand(M_SEED);
    for (int i = 0; i < W * H; i++) h_in[i] = rand() % 10;

    int *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
    stencil2d_kernel<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    stencil2d_cpu(h_in, h_ref, W, H);

    int mismatches = 0;
    for (int i = 0; i < W * H; i++) {
        if (h_out[i] != h_ref[i]) {
            if (mismatches < 8)
                printf("Mismatch at (%d,%d): GPU=%d CPU=%d\n", i % W, i / W, h_out[i], h_ref[i]);
            mismatches++;
        }
    }
    printf("stencil2d: %s (%dx%d, %d mismatch%s).\n",
           mismatches == 0 ? "CPU and GPU match" : "FAIL",
           W, H, mismatches, mismatches == 1 ? "" : "es");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);

    return mismatches == 0 ? 0 : 1;
}
