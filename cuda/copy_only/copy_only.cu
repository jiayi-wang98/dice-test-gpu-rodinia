#include <stdio.h>
#include <stdlib.h>
__global__ void copy_only_kernel(const int *in, int *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] + 1;
}
int main(int argc, char **argv) {
    int N = (argc>1) ? atoi(argv[1]) : 256;
    int *h_in = (int*)malloc(N*sizeof(int));
    int *h_out = (int*)malloc(N*sizeof(int));
    for (int i=0; i<N; i++) h_in[i] = i;
    int *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(int)); cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);
    copy_only_kernel<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    int bad=0;
    for (int i=0; i<N; i++) if (h_out[i] != h_in[i]+1) bad++;
    printf("copy_only: %s (%d/%d mismatches)\n", bad==0?"OK":"FAIL", bad, N);
    return bad==0 ? 0 : 1;
}
