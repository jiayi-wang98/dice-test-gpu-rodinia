// stencil3d: 7-point 3-D Jacobi over an int field.
// out[i,j,k] = in[i,j,k] + in[i-1,j,k] + in[i+1,j,k]
//            + in[i,j-1,k] + in[i,j+1,k]
//            + in[i,j,k-1] + in[i,j,k+1]
// Boundary: replicate (clamp).
//
// Block layout: BLOCK_X x BLOCK_Y x BLOCK_Z = 8 x 8 x 4 = 256 threads.
// SMEM tile is 8x8x4 ints = 1 KiB.
//
// 6 SMEM neighbour reads -> exceeds the 4-port budget in the RF-unified
// design; the partition pass splits the compute p-graph into two
// (horizontal pass, vertical pass).  See stencil3d.1.sm_52.rfu.{meta,pptx}.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BX 8
#define BY 8
#define BZ 4
#define DEVICE 0
#define M_SEED 9

__global__ void stencil3d_kernel(const int *in, int *out, int W, int H, int D)
{
    __shared__ int tile[BZ][BY][BX];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int gx = blockIdx.x * BX + tx;
    int gy = blockIdx.y * BY + ty;
    int gz = blockIdx.z * BZ + tz;

    int v = (gx < W && gy < H && gz < D) ? in[(gz * H + gy) * W + gx] : 0;
    tile[tz][ty][tx] = v;
    __syncthreads();

    if (gx >= W || gy >= H || gz >= D) return;

    int center = tile[tz][ty][tx];

    int e = (tx < BX-1) ? tile[tz][ty][tx+1] : ((gx < W-1) ? in[(gz*H + gy)*W + (gx+1)] : center);
    int w = (tx > 0)    ? tile[tz][ty][tx-1] : ((gx > 0)   ? in[(gz*H + gy)*W + (gx-1)] : center);
    int n = (ty < BY-1) ? tile[tz][ty+1][tx] : ((gy < H-1) ? in[(gz*H + (gy+1))*W + gx] : center);
    int s = (ty > 0)    ? tile[tz][ty-1][tx] : ((gy > 0)   ? in[(gz*H + (gy-1))*W + gx] : center);
    int t = (tz < BZ-1) ? tile[tz+1][ty][tx] : ((gz < D-1) ? in[((gz+1)*H + gy)*W + gx] : center);
    int b = (tz > 0)    ? tile[tz-1][ty][tx] : ((gz > 0)   ? in[((gz-1)*H + gy)*W + gx] : center);

    out[(gz * H + gy) * W + gx] = center + e + w + n + s + t + b;
}

void stencil3d_cpu(const int *in, int *out, int W, int H, int D)
{
    for (int k = 0; k < D; k++)
    for (int j = 0; j < H; j++)
    for (int i = 0; i < W; i++) {
        int c = in[(k*H + j)*W + i];
        int e = (i < W-1) ? in[(k*H + j)*W + (i+1)] : c;
        int w = (i > 0)   ? in[(k*H + j)*W + (i-1)] : c;
        int n = (j < H-1) ? in[(k*H + (j+1))*W + i] : c;
        int s = (j > 0)   ? in[(k*H + (j-1))*W + i] : c;
        int t = (k < D-1) ? in[((k+1)*H + j)*W + i] : c;
        int b = (k > 0)   ? in[((k-1)*H + j)*W + i] : c;
        out[(k*H + j)*W + i] = c + e + w + n + s + t + b;
    }
}

int main(int argc, char **argv)
{
    int W = (argc > 1) ? atoi(argv[1]) : 32;
    int H = (argc > 2) ? atoi(argv[2]) : 32;
    int D = (argc > 3) ? atoi(argv[3]) : 16;

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    size_t bytes = (size_t)W * H * D * sizeof(int);
    int *h_in   = (int *)malloc(bytes);
    int *h_out  = (int *)malloc(bytes);
    int *h_ref  = (int *)malloc(bytes);

    srand(M_SEED);
    for (int i = 0; i < W * H * D; i++) h_in[i] = rand() % 10;

    int *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(BX, BY, BZ);
    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, (D + BZ - 1) / BZ);
    stencil3d_kernel<<<grid, block>>>(d_in, d_out, W, H, D);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    stencil3d_cpu(h_in, h_ref, W, H, D);

    int mismatches = 0;
    for (int i = 0; i < W * H * D; i++) {
        if (h_out[i] != h_ref[i]) {
            if (mismatches < 8)
                printf("Mismatch at %d: GPU=%d CPU=%d\n", i, h_out[i], h_ref[i]);
            mismatches++;
        }
    }
    printf("stencil3d: %s (%dx%dx%d, %d mismatch%s).\n",
           mismatches == 0 ? "CPU and GPU match" : "FAIL",
           W, H, D, mismatches, mismatches == 1 ? "" : "es");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);

    return mismatches == 0 ? 0 : 1;
}
