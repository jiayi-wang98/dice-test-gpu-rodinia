#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
struct timeval tv_cpu_start, tv_cpu_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, cpu_time = 0;
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT // Enable for debugging

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
int* result_cpu;
#define M_SEED 9
int pyramid_height;
static int max_iterations = 1; // Allow all iterations
static int cuda_kernel_called_times = 0;

void
init(int argc, char** argv)
{
	if(argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	result_cpu = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    printf("Input wall array:\n");
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", wall[i][j]);
        }
        printf("\n");
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResult,
                int cols, 
                int rows,
                int startStep,
                int border)
{
        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
        int small_block_cols = BLOCK_SIZE-iteration*HALO*2;
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        int xidx = blkX+tx;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads();
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)
                prev[tx]= result[tx];
	    __syncthreads();
      }

      if (computed){
          gpuResult[xidx]=result[tx];		
      }
}

/*
   Compute partial time steps on CPU for one GPU kernel launch
*/
void calc_path_cpu_partial(int *wall, int *src, int *dst, int cols, int rows, int startStep, int iterations, int border)
{
    int small_block_cols = BLOCK_SIZE - iterations * HALO * 2;
    
    // Process each block
    for (int bx = 0; bx < (cols + small_block_cols - 1) / small_block_cols; bx++) {
        int blkX = small_block_cols * bx - border;
        int blkXmax = blkX + BLOCK_SIZE - 1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
        
        // Allocate block-local arrays to simulate GPU shared memory
        int prev[BLOCK_SIZE];
        int result[BLOCK_SIZE];
        
        // Initialize arrays (important for out-of-bounds accesses)
        for (int tx = 0; tx < BLOCK_SIZE; tx++) {
            prev[tx] = 0;
            result[tx] = 0;
        }
        
        // Load source data into prev array
        for (int tx = 0; tx < BLOCK_SIZE; tx++) {
            int xidx = blkX + tx;
            if (IN_RANGE(xidx, 0, cols-1)) {
                prev[tx] = src[xidx];
            }
        }
        
        // Process iterations within this kernel call
        for (int i = 0; i < iterations; i++) {
            // Compute for all threads in block
            for (int tx = 0; tx < BLOCK_SIZE; tx++) {
                int xidx = blkX + tx;
                bool isValid = IN_RANGE(tx, validXmin, validXmax);
                
                if (IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid && IN_RANGE(xidx, 0, cols-1)) {
                    int W = tx - 1;
                    int E = tx + 1;
                    W = (W < validXmin) ? validXmin : W;
                    E = (E > validXmax) ? validXmax : E;
                    
                    int left = prev[W];
                    int up = prev[tx];
                    int right = prev[E];
                    int shortest = MIN(left, up);
                    shortest = MIN(shortest, right);
                    int index = cols * (startStep + i) + xidx;
                    result[tx] = shortest + wall[index];
                }
            }
            
            // Copy result to prev for next iteration (if not last iteration)
            if (i < iterations - 1) {
                for (int tx = 0; tx < BLOCK_SIZE; tx++) {
                    int xidx = blkX + tx;
                    bool isValid = IN_RANGE(tx, validXmin, validXmax);
                    // Only copy computed values
                    if (IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid && IN_RANGE(xidx, 0, cols-1)) {
                        prev[tx] = result[tx];
                    }
                }
            }
        }
        
        // Write final results back to dst
        for (int tx = 0; tx < BLOCK_SIZE; tx++) {
            int xidx = blkX + tx;
            bool isValid = IN_RANGE(tx, validXmin, validXmax);
            int final_i = iterations - 1;
            if (IN_RANGE(tx, final_i+1, BLOCK_SIZE-final_i-2) && isValid && IN_RANGE(xidx, 0, cols-1)) {
                dst[xidx] = result[tx];
            }
        }
    }
}
/*
   Compute N time steps on GPU with per-kernel comparison
*/
int calc_path(int *gpuWall, int *gpuResult[2], int *cpuSrc, int *cpuDst, int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);  
    
    int src = 1, dst = 0;
    
    for (int t = 0; t < rows-1; t+=pyramid_height) {
        int iterations = MIN(pyramid_height, rows-t-1);
        
        // Run GPU kernel
#ifdef TIMING
        gettimeofday(&tv_kernel_start, NULL);
#endif
        // Swap GPU buffers
        int temp = src;
        src = dst;
        dst = temp;
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            iterations, 
            gpuWall, gpuResult[src], gpuResult[dst],
            cols, rows, t, borderCols);
        cudaDeviceSynchronize();
        cuda_kernel_called_times++;
        
#ifdef TIMING
        gettimeofday(&tv_kernel_end, NULL);
        tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
        kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

        // Copy GPU results to host
        cudaMemcpy(result, gpuResult[dst], sizeof(int)*cols, cudaMemcpyDeviceToHost);
        
        // Run CPU computation
#ifdef TIMING
        gettimeofday(&tv_cpu_start, NULL);
#endif
        
        // For the first iteration, GPU reads from gpuResult[0] (after swap src=0)
        // So CPU should also read from the buffer with initial data
        // For subsequent iterations, we alternate
        int *cpuSrcPtr, *cpuDstPtr;
        
        // Match the GPU's buffer pattern exactly
        if (src == 0) {
            // GPU is reading from buffer 0, writing to buffer 1
            cpuSrcPtr = cpuSrc;  // cpuSrc has the initial data, like gpuResult[0]
            cpuDstPtr = cpuDst;
        } else {
            // GPU is reading from buffer 1, writing to buffer 0
            cpuSrcPtr = cpuDst;
            cpuDstPtr = cpuSrc;
        }
        
        calc_path_cpu_partial(data+cols, cpuSrcPtr, cpuDstPtr, cols, rows, t, iterations, borderCols);
        
#ifdef TIMING
        gettimeofday(&tv_cpu_end, NULL);
        tvsub(&tv_cpu_end, &tv_cpu_start, &tv);
        cpu_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

        // Compare results
        bool match = true;
        for (int i = 0; i < cols; i++) {
            if (result[i] != cpuDstPtr[i]) {
                match = false;
                printf("Mismatch at iteration %d, index %d: GPU = %d, CPU = %d\n", 
                       cuda_kernel_called_times, i, result[i], cpuDstPtr[i]);
                break;
            }
        }
        printf("Iteration %d: CPU and GPU results %s\n", cuda_kernel_called_times, match ? "match" : "do not match");

#ifdef BENCH_PRINT
        printf("GPU results after iteration %d:\n", cuda_kernel_called_times);
        for (int i = 0; i < cols; i++)
            printf("%d ", result[i]);
        printf("\nCPU results after iteration %d:\n", cuda_kernel_called_times);
        for (int i = 0; i < cols; i++)
            printf("%d ", cpuDstPtr[i]);
        printf("\n");
#endif

            if (!match) {
	    	    printf("[**ERROR] GPU and CPU results are different in the kernel!!\n");
	    	    printf("CPU and GPU results differ!\n");
	    	    printf("\n");
	    	    printf("**        **\n");
	    	    printf(" **      ** \n");
	    	    printf("  **    **  \n");
	    	    printf("   **  **   \n");
	    	    printf("   **  **   \n");
	    	    printf("  **    **  \n");
	    	    printf(" **      ** \n");
	    	    printf("**        **\n");
	    	    printf("\n");
	    	    // Flush output buffers
	    	    fflush(stdout);
	    	    fflush(stderr);
	    	    exit(EXIT_FAILURE);
	        } else {
	        	printf("CPU and GPU results match!\n");
	        	printf("\n");
	        	printf("       .-\"\"\"\"\"-.\n");
	        	printf("     .'         '.\n");
	        	printf("    :             :\n");
	        	printf("   :    ^     ^    :\n");
	        	printf("   :     .---.     :\n");
	        	printf("    :   (     )   :\n");
	        	printf("     '.  '---'  .'\n");
	        	printf("       '-.....-'\n");
	        	printf("\n");
	        }

// Copy CPU results to GPU for next iteration
        cudaMemcpy(gpuResult[dst], cpuDstPtr, sizeof(int)*cols, cudaMemcpyHostToDevice);

        if (cuda_kernel_called_times >= max_iterations) {
            break;
        }
    }
    return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int *cpuSrc, *cpuDst;
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

    cpuSrc = new int[cols];
    cpuDst = new int[cols];
    for (int i = 0; i < cols; i++) {
        cpuSrc[i] = data[i];
    }

#ifdef BENCH_PRINT
    printf("Initial src (CPU and GPU):\n");
    for (int i = 0; i < cols; i++)
        printf("%d ", cpuSrc[i]);
    printf("\n");
#endif

    int final_ret = calc_path(gpuWall, gpuResult, cpuSrc, cpuDst, rows, cols, \
	 pyramid_height, blockCols, borderCols);

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);

#ifdef TIMING
    printf("CPU Total Exec: %f ms\n", cpu_time);
    printf("GPU Total Exec: %f ms\n", kernel_time);
#endif

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);
    delete [] cpuSrc;
    delete [] cpuDst;
    delete [] data;
    delete [] wall;
    delete [] result;
    delete [] result_cpu;
}