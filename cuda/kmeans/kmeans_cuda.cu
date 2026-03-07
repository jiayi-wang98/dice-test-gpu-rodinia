#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include <omp.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "kmeans_cuda_kernel.cu"


//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

extern "C"
int setup(int argc, char** argv);									/* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */

static int g_invert_verify_runs = 0;
static int g_kmeanspoint_verify_runs = 0;

static int
verify_invert_mapping_cpu(const float *row_major,
						  const float *column_major,
						  int npoints,
						  int nfeatures)
{
	int p;
	int f;
	int mismatch_count = 0;
	int printed = 0;
	const int max_print = 5;

	for (p = 0; p < npoints; p++) {
		for (f = 0; f < nfeatures; f++) {
			float expected = row_major[p * nfeatures + f];
			float got = column_major[p + npoints * f];
			if (expected != got) {
				mismatch_count++;
				if (printed < max_print) {
					printf("[CPU-VERIFY][invert_mapping] mismatch point=%d feature=%d gpu=%f cpu=%f\n",
						   p, f, got, expected);
					printed++;
				}
			}
		}
	}

	return mismatch_count;
}

static int
verify_kmeanspoint_membership_cpu(float **feature,
								  int nfeatures,
								  int npoints,
								  int nclusters,
								  float **clusters,
								  const int *membership_gpu)
{
	int i;
	int j;
	int c;
	int mismatch_count = 0;
	int printed = 0;
	const int max_print = 5;

	for (i = 0; i < npoints; i++) {
		float min_dist = FLT_MAX;
		int expected_cluster = -1;

		for (c = 0; c < nclusters; c++) {
			float dist = 0.0f;
			for (j = 0; j < nfeatures; j++) {
				float diff = feature[i][j] - clusters[c][j];
				dist += diff * diff;
			}
			if (dist < min_dist) {
				min_dist = dist;
				expected_cluster = c;
			}
		}

		if (membership_gpu[i] != expected_cluster) {
			mismatch_count++;
			if (printed < max_print) {
				printf("[CPU-VERIFY][kmeansPoint] mismatch point=%d gpu=%d cpu=%d\n",
					   i, membership_gpu[i], expected_cluster);
				printed++;
			}
		}
	}

	return mismatch_count;
}


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{	
	num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double) num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
		num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*) malloc(npoints * sizeof(int));
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
	
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
	cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));
		
	/* invert the data array (kernel execution) */	
	invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);
	cudaThreadSynchronize();

	float *feature_inverted_h = (float *) malloc(npoints * nfeatures * sizeof(float));
	cudaMemcpy(feature_inverted_h, feature_d, npoints * nfeatures * sizeof(float), cudaMemcpyDeviceToHost);
	g_invert_verify_runs++;
	int invert_mismatch = verify_invert_mapping_cpu(features[0],
														feature_inverted_h,
														npoints,
														nfeatures);
	if (invert_mismatch == 0) {
		printf("[CPU-VERIFY] invert_mapping run %d: PASS (%d x %d)\n",
			   g_invert_verify_runs,
			   npoints,
			   nfeatures);
	} else {
		printf("[CPU-VERIFY] invert_mapping run %d: FAIL (%d mismatches)\n",
			   g_invert_verify_runs,
			   invert_mismatch);
		exit(EXIT_FAILURE);
	}
	free(feature_inverted_h);
		
	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	cudaMalloc((void**) &membership_d, npoints*sizeof(int));
	cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));

	
#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side
	
	cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
	//cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
	cudaMalloc((void**) &block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
	//cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
	cudaFree(feature_d);
	cudaFree(feature_flipped_d);
	cudaFree(membership_d);

	cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
    cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
    cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */



////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
	// make sure we're running on the big card
    cudaSetDevice(1);
	// as done in the CUDA start/help document provided
	setup(argc, argv);    
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ------------------- kmeansCuda() ------------------------ */    
extern "C"
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers				/* sum of elements in each cluster */
		   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */


	cudaSetDevice(1);

	/* copy membership (host to device) */
	cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);

	/* copy clusters (host to device) */
	cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);

    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );
    
	/* execute the kernel */
    kmeansPoint<<< grid, threads >>>( feature_d,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership_d,
                                      clusters_d,
									  block_clusters_d,
									  block_deltas_d);

	cudaThreadSynchronize();

	/* copy back membership (device to host) */
	cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);	

	g_kmeanspoint_verify_runs++;
	int kmeanspoint_mismatch = verify_kmeanspoint_membership_cpu(feature,
															 nfeatures,
															 npoints,
															 nclusters,
															 clusters,
															 membership_new);
	if (kmeanspoint_mismatch == 0) {
		printf("[CPU-VERIFY] kmeansPoint iter %d: PASS (%d points)\n",
			   g_kmeanspoint_verify_runs,
			   npoints);
	} else {
		printf("[CPU-VERIFY] kmeansPoint iter %d: FAIL (%d mismatches)\n",
			   g_kmeanspoint_verify_runs,
			   kmeanspoint_mismatch);
		exit(EXIT_FAILURE);
	}

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
	cudaMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
        
	cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost);
#endif
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	

#ifdef BLOCK_DELTA_REDUCE	
    /*** calculate global sums from per block sums for delta and the new centers ***/    
	
	//debug
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }
        
#endif
#ifdef BLOCK_CENTER_REDUCE	
	
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for(int j = 0; j < nclusters;j++) {
			for(int k = 0; k < nfeatures;k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
    }
	

#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
				printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
			}
		}
	}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++)
			new_centers[j][k]= block_new_centers[j*nfeatures + k];		
	}
#endif

#endif

	return delta;
	
}
/* ------------------- kmeansCuda() end ------------------------ */    

