// 5-point stencil, in-block only (no cross-block gmem fallback).
#include <stdio.h>
#include <stdlib.h>
__global__ void k(const int *in, int *out, int W, int H) {
    __shared__ int tile[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * 16 + tx;
    int gy = blockIdx.y * 16 + ty;
    int v = (gx < W && gy < H) ? in[gy*W + gx] : 0;
    tile[ty][tx] = v;
    __syncthreads();
    if (gx >= W || gy >= H) return;
    int center = tile[ty][tx];
    int up    = (ty > 0)         ? tile[ty-1][tx] : center;
    int down  = (ty < 15)        ? tile[ty+1][tx] : center;
    int left  = (tx > 0)         ? tile[ty][tx-1] : center;
    int right = (tx < 15)        ? tile[ty][tx+1] : center;
    out[gy*W + gx] = center + up + down + left + right;
}
int main() {
    int W = 16, H = 16;
    size_t bytes = W * H * sizeof(int);
    int *h_in=(int*)malloc(bytes), *h_out=(int*)malloc(bytes), *ref=(int*)malloc(bytes);
    for (int i = 0; i < W*H; i++) h_in[i] = i % 7 + 1;
    for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
        int c = h_in[j*W + i];
        int u = (j > 0) ? h_in[(j-1)*W + i] : c;
        int d = (j < H-1) ? h_in[(j+1)*W + i] : c;
        int l = (i > 0) ? h_in[j*W + (i-1)] : c;
        int r = (i < W-1) ? h_in[j*W + (i+1)] : c;
        ref[j*W + i] = c + u + d + l + r;
    }
    int *d_in,*d_out; cudaMalloc(&d_in,bytes); cudaMalloc(&d_out,bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 b(16, 16); dim3 g(1, 1);
    k<<<g, b>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < W*H; i++) if (h_out[i] != ref[i]) {
        if (bad < 5) printf("mismatch at %d (j=%d,i=%d): GPU=%d ref=%d\n", i, i/W, i%W, h_out[i], ref[i]);
        bad++;
    }
    printf("stencil2d_5pt: %s (%d/%d wrong)\n", bad==0?"OK":"FAIL", bad, W*H);
    return bad?1:0;
}
