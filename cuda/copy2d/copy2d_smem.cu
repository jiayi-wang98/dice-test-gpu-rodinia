#include <stdio.h>
#include <stdlib.h>
__global__ void copy2d_smem_kernel(const int *in, int *out, int W, int H) {
    __shared__ int tile[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * 16 + tx;
    int gy = blockIdx.y * 16 + ty;
    int v = (gx < W && gy < H) ? in[gy*W + gx] : 0;
    tile[ty][tx] = v;
    __syncthreads();
    if (gx < W && gy < H) out[gy*W + gx] = tile[ty][tx] + 1;
}
int main(int argc, char **argv) {
    int W = (argc>1) ? atoi(argv[1]) : 16;
    int H = (argc>2) ? atoi(argv[2]) : 16;
    size_t bytes = W * H * sizeof(int);
    int *h_in = (int*)malloc(bytes), *h_out = (int*)malloc(bytes);
    for (int i = 0; i < W*H; i++) h_in[i] = i;
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 b(16, 16); dim3 g((W+15)/16, (H+15)/16);
    copy2d_smem_kernel<<<g, b>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < W*H; i++) if (h_out[i] != h_in[i]+1) bad++;
    printf("copy2d_smem: %s (%dx%d, %d/%d wrong)\n", bad==0?"OK":"FAIL", W, H, bad, W*H);
    return bad?1:0;
}
