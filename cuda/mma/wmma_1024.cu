#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define M 1024
#define N 1024
#define K 1024

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_PER_BLOCK_ROWS 8
#define WARPS_PER_BLOCK_COLS 4
#define WARPS_PER_BLOCK (WARPS_PER_BLOCK_ROWS * WARPS_PER_BLOCK_COLS)

__global__ void wmma_gemm_1024x1024(const half *A, const half *B, float *C) {
    extern __shared__ half shmem[];
    half *As = shmem; 
    half *Bs = As + (WARPS_PER_BLOCK_ROWS * WMMA_M * WMMA_K);

    const int block_row = blockIdx.y * (WARPS_PER_BLOCK_ROWS * WMMA_M); // 128
    const int block_col = blockIdx.x * (WARPS_PER_BLOCK_COLS * WMMA_N); // 64

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warp_row = warpId / WARPS_PER_BLOCK_COLS; 
    const int warp_col = warpId % WARPS_PER_BLOCK_COLS;

    const int c_row = block_row + warp_row * WMMA_M;
    const int c_col = block_col + warp_col * WMMA_N;

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // shared tile sizes
    const int A_sh_rows = WARPS_PER_BLOCK_ROWS * WMMA_M; // 128
    const int A_sh_cols = WMMA_K;                        // 16
    const int B_sh_rows = WMMA_K;                        // 16
    const int B_sh_cols = WARPS_PER_BLOCK_COLS * WMMA_N; // 64
    const int As_elems = A_sh_rows * A_sh_cols;           // 2048
    const int Bs_elems = B_sh_rows * B_sh_cols;           // 1024

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // load A tile (128x16)
        for (int idx = tid; idx < As_elems; idx += blockDim.x) {
            int r = idx / A_sh_cols;
            int c = idx % A_sh_cols;
            int g_r = block_row + r;
            int g_c = k0 + c;
            As[idx] = A[g_r * K + g_c];
        }
        // load B tile (16x64)
        for (int idx = tid; idx < Bs_elems; idx += blockDim.x) {
            int r = idx / B_sh_cols;
            int c = idx % B_sh_cols;
            int g_r = k0 + r;
            int g_c = block_col + c;
            Bs[idx] = B[g_r * N + g_c];
        }

        __syncthreads();

        half const *tileA = &As[(warp_row * WMMA_M) * A_sh_cols + 0];
        half const *tileB = &Bs[0 * B_sh_cols + (warp_col * WMMA_N)];

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, tileA, A_sh_cols);
        wmma::load_matrix_sync(b_frag, tileB, B_sh_cols);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // store accumulator fragment to global C
    wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag, N, wmma::mem_row_major);
}

// ---------------- MAIN -----------------

int main() {
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    half *hA = (half*)malloc(size_A);
    half *hB = (half*)malloc(size_B);
    float *hC = (float*)malloc(size_C);

    for (int i = 0; i < M*K; i++) hA[i] = __float2half((float)(rand()%3));
    for (int i = 0; i < K*N; i++) hB[i] = __float2half((float)(rand()%3));

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, size_A);
    cudaMalloc(&dB, size_B);
    cudaMalloc(&dC, size_C);
    cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, size_C);

    dim3 grid(N / (WARPS_PER_BLOCK_COLS*WMMA_N), M / (WARPS_PER_BLOCK_ROWS*WMMA_M));
    dim3 block(WARPS_PER_BLOCK * 32); // 1024 threads
    size_t shmemBytes = (128*16 + 16*64)*sizeof(half); // 2048+1024 = 3072 elements

    wmma_gemm_1024x1024<<<grid, block, shmemBytes>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

    printf("Done. Sample C[0]=%f\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
