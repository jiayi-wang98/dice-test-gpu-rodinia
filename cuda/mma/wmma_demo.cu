#include <mma.h>          // for WMMA API
#include <cuda_fp16.h>    // for __half
#include <cstdio>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void wmma_m16n16k16_fp16_fp32(const half *A, const half *B, float *C) {
    // Fragments are register containers for submatrices
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Each warp loads the whole 16x16 tile (one warp processes one tile)
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, K);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}

int main() {
    half *d_A, *d_B;
    float *d_C;
    half h_A[M*K], h_B[K*N];
    float h_C[M*N];

    // Initialize host matrices
    for (int i = 0; i < M*K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half(1.0f);

    cudaMalloc(&d_A, M*K*sizeof(half));
    cudaMalloc(&d_B, K*N*sizeof(half));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice);

    // One warp is enough (32 threads)
    wmma_m16n16k16_fp16_fp32<<<1, 32>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.1f ", h_C[i*N+j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}