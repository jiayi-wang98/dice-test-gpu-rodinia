// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
int verify_array_mismatch(const char *kernel_name,
                          const char *array_name,
                          const float *cpu_data,
                          const float *gpu_data,
                          int size,
                          int iter,
                          float abs_tolerance,
                          float rel_tolerance);
void print_large_smile();
void print_large_x();
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}


void
runTest( int argc, char** argv) 
{
    int rows, cols, size_I, size_R, niter = 10, iter, k;
    int verification_failed = 0;
    float *I, *J, *J_cpu, lambda;
    float q0sqr_gpu, q0sqr_cpu;
    float sum, sum2, tmp, meanROI, varROI;
    float sum_cpu, sum2_cpu, tmp_cpu, meanROI_cpu, varROI_cpu;
    const float abs_tolerance = 1.0e-4f;
    const float rel_tolerance = 1.0e-3f;

	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;

#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;
    float *C_gpu, *E_gpu, *W_gpu, *N_gpu, *S_gpu;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
    J_cpu = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


    iN = (int *)malloc(sizeof(int) * rows) ;
    iS = (int *)malloc(sizeof(int) * rows) ;
    jW = (int *)malloc(sizeof(int) * cols) ;
    jE = (int *)malloc(sizeof(int) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#ifdef GPU

	//Allocate device memory
    cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
    cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
	cudaMalloc((void**)& E_C, sizeof(float)* size_I);
	cudaMalloc((void**)& W_C, sizeof(float)* size_I);
	cudaMalloc((void**)& S_C, sizeof(float)* size_I);
	cudaMalloc((void**)& N_C, sizeof(float)* size_I);
    C_gpu = (float *)malloc(sizeof(float) * size_I);
    E_gpu = (float *)malloc(sizeof(float) * size_I);
    W_gpu = (float *)malloc(sizeof(float) * size_I);
    N_gpu = (float *)malloc(sizeof(float) * size_I);
    S_gpu = (float *)malloc(sizeof(float) * size_I);

	
#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
        J_cpu[k] = J[k];
    }
	printf("Start the SRAD main loop\n");
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr_gpu = varROI / (meanROI*meanROI);

		sum_cpu = 0;
        sum2_cpu = 0;
        for (int i = r1; i <= r2; i++) {
            for (int j = c1; j <= c2; j++) {
                tmp_cpu = J_cpu[i * cols + j];
                sum_cpu += tmp_cpu;
                sum2_cpu += tmp_cpu * tmp_cpu;
            }
        }
        meanROI_cpu = sum_cpu / size_R;
        varROI_cpu  = (sum2_cpu / size_R) - meanROI_cpu * meanROI_cpu;
        q0sqr_cpu   = varROI_cpu / (meanROI_cpu * meanROI_cpu);

		for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J_cpu[k];
 
				// directional derivates
                dN[k] = J_cpu[iN[i] * cols + j] - Jc;
                dS[k] = J_cpu[iS[i] * cols + j] - Jc;
                dW[k] = J_cpu[i * cols + jW[j]] - Jc;
                dE[k] = J_cpu[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr_cpu) / (q0sqr_cpu * (1+q0sqr_cpu)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J_cpu[k] = J_cpu[k] + 0.25*lambda*D;
            }
	}

#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    

	//Copy data from main memory to device memory
	cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr_gpu); 

    // Verify kernel-1 outputs before running kernel-2.
    cudaMemcpy(C_gpu, C_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(E_gpu, E_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(W_gpu, W_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(N_gpu, N_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
    cudaMemcpy(S_gpu, S_C, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

    if (verify_array_mismatch("srad_cuda_1", "C_cuda", c, C_gpu, size_I, iter, abs_tolerance, rel_tolerance) ||
        verify_array_mismatch("srad_cuda_1", "E_C", dE, E_gpu, size_I, iter, abs_tolerance, rel_tolerance) ||
        verify_array_mismatch("srad_cuda_1", "W_C", dW, W_gpu, size_I, iter, abs_tolerance, rel_tolerance) ||
        verify_array_mismatch("srad_cuda_1", "N_C", dN, N_gpu, size_I, iter, abs_tolerance, rel_tolerance) ||
        verify_array_mismatch("srad_cuda_1", "S_C", dS, S_gpu, size_I, iter, abs_tolerance, rel_tolerance)) {
        verification_failed = 1;
        printf("Stopping immediately due to kernel-1 mismatch.\n");
        print_large_x();
        goto cleanup;
    }

    srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr_gpu); 

	//Copy data from device memory to main memory
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

    if (verify_array_mismatch("srad_cuda_2", "J_cuda", J_cpu, J, size_I, iter, abs_tolerance, rel_tolerance)) {
        verification_failed = 1;
        printf("Stopping immediately due to kernel-2 mismatch.\n");
        print_large_x();
        goto cleanup;
    }

#endif   
}

    cudaThreadSynchronize();

    if (!verification_failed) {
        printf("CPU vs GPU verification: PASS (all kernel launches matched)\n");
        print_large_smile();
    }

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");

cleanup:
	free(I);
	free(J);
    free(J_cpu);
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#ifdef GPU
    cudaFree(C_cuda);
	cudaFree(J_cuda);
	cudaFree(E_C);
	cudaFree(W_C);
	cudaFree(N_C);
	cudaFree(S_C);
    free(C_gpu);
    free(E_gpu);
    free(W_gpu);
    free(N_gpu);
    free(S_gpu);
#endif 
	free(c);

    if (verification_failed) {
        exit(EXIT_FAILURE);
    }
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}


int verify_array_mismatch(const char *kernel_name,
                          const char *array_name,
                          const float *cpu_data,
                          const float *gpu_data,
                          int size,
                          int iter,
                          float abs_tolerance,
                          float rel_tolerance){
    int mismatches = 0;
    int first_index = -1;
    int max_index = 0;
    float first_cpu_value = 0.0f;
    float first_gpu_value = 0.0f;
    float first_abs_diff = 0.0f;
    float max_abs_diff = 0.0f;

    for (int idx = 0; idx < size; idx++) {
        float cpu_value = cpu_data[idx];
        float gpu_value = gpu_data[idx];
        float abs_diff = fabsf(cpu_value - gpu_value);
        float scale = fmaxf(fabsf(cpu_value), fabsf(gpu_value));

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_index = idx;
        }

        if ((abs_diff > abs_tolerance) && (abs_diff > rel_tolerance * scale)) {
            if (first_index < 0) {
                first_index = idx;
                first_cpu_value = cpu_value;
                first_gpu_value = gpu_value;
                first_abs_diff = abs_diff;
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        printf("CPU vs GPU verification: FAIL at iteration %d after %s on %s\n",
               iter + 1, kernel_name, array_name);
        printf("  mismatches=%d, first_idx=%d, cpu=%.8f, gpu=%.8f, abs_diff=%.8f\n",
               mismatches, first_index, first_cpu_value, first_gpu_value, first_abs_diff);
        printf("  max_abs_diff=%.8f at idx=%d\n", max_abs_diff, max_index);
        return 1;
    }

    return 0;
}


void print_large_smile(){
    printf("\n");
    printf("  ****************************\n");
    printf(" *                            *\n");
    printf("*      ***            ***      *\n");
    printf("*      ***            ***      *\n");
    printf("*                              *\n");
    printf("*        ****************      *\n");
    printf("*          ************        *\n");
    printf(" *                            *\n");
    printf("  ****************************\n");
    printf("\n");
}

void print_large_x(){
    printf("\n");
    printf("XX                      XX\n");
    printf("  XX                  XX\n");
    printf("    XX              XX\n");
    printf("      XX          XX\n");
    printf("        XX      XX\n");
    printf("          XX  XX\n");
    printf("          XX  XX\n");
    printf("        XX      XX\n");
    printf("      XX          XX\n");
    printf("    XX              XX\n");
    printf("  XX                  XX\n");
    printf("XX                      XX\n");
    printf("\n");
}

