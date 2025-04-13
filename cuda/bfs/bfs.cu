/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#ifdef TIMING
#include "timing.h"
#endif

#define MAX_THREADS_PER_BLOCK 512

int no_of_nodes;
int edge_list_size;
FILE *fp;

#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

// Add this before the main BFSGraph function
void BFS_CPU(Node* graph_nodes, int* graph_edges, bool* graph_visited, 
             int* cost, int no_of_nodes, int source) {
    // Simple queue implementation for CPU BFS
    int* queue = (int*)malloc(sizeof(int) * no_of_nodes);
    int front = 0, rear = 0;
    
    // Initialize cost array
    for(int i = 0; i < no_of_nodes; i++) {
        cost[i] = -1;
        graph_visited[i] = false;
    }
    
    // Start with source node
    cost[source] = 0;
    graph_visited[source] = true;
    queue[rear++] = source;
    
    while(front < rear) {
        int current = queue[front++];
        
        // Process all neighbors
        int start = graph_nodes[current].starting;
        int end = start + graph_nodes[current].no_of_edges;
        
        for(int i = start; i < end; i++) {
            int neighbor = graph_edges[i];
            if(!graph_visited[neighbor]) {
                cost[neighbor] = cost[current] + 1;
                graph_visited[neighbor] = true;
                queue[rear++] = neighbor;
            }
        }
    }
    
    free(queue);
}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
// Modified BFSGraph function with CPU comparison
void BFSGraph(int argc, char** argv) 
{
    char *input_f;
    if(argc!=2){
        Usage(argc, argv);
        exit(0);
    }

    input_f = argv[1];
    printf("Reading File\n");
    //Read in Graph from a file
    fp = fopen(input_f,"r");
    if(!fp)
    {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;
    fscanf(fp,"%d",&no_of_nodes);

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    if(no_of_nodes>MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }

    // allocate host memory
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited_cpu = (bool*) malloc(sizeof(bool)*no_of_nodes);

    int start, edgeno;   
    for(unsigned int i = 0; i < no_of_nodes; i++) 
    {
        fscanf(fp,"%d %d",&start,&edgeno);
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]=false;
        h_updating_graph_mask[i]=false;
        h_graph_visited[i]=false;
        h_graph_visited_cpu[i]=false;
    }

    fscanf(fp,"%d",&source);
    source=0;

    h_graph_mask[source]=true;
    h_graph_visited[source]=true;

    fscanf(fp,"%d",&edge_list_size);

    int id,cost;
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    for(int i=0; i < edge_list_size ; i++)
    {
        fscanf(fp,"%d",&id);
        fscanf(fp,"%d",&cost);
        h_graph_edges[i] = id;
    }

    if(fp)
        fclose(fp);    

    printf("Read File\n");

    // Allocate memory for CPU result
    int* h_cost_cpu = (int*) malloc(sizeof(int)*no_of_nodes);

    // Run CPU version
    printf("Running CPU BFS\n");
    BFS_CPU(h_graph_nodes, h_graph_edges, h_graph_visited_cpu, h_cost_cpu, no_of_nodes, source);

    #ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
    #endif

    //Copy the Node list to device memory
    Node* d_graph_nodes;
    cudaMalloc((void**) &d_graph_nodes, sizeof(Node)*no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice);

    //Copy the Edge List to device Memory
    int* d_graph_edges;
    cudaMalloc((void**) &d_graph_edges, sizeof(int)*edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);

    //Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc((void**) &d_graph_mask, sizeof(bool)*no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

    bool* d_updating_graph_mask;
    cudaMalloc((void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc((void**) &d_graph_visited, sizeof(bool)*no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

    // allocate mem for the result on host side
    int* h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
    for(int i=0;i<no_of_nodes;i++)
        h_cost[i]=-1;
    h_cost[source]=0;
    
    // allocate device memory for result
    int* d_cost;
    cudaMalloc((void**) &d_cost, sizeof(int)*no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

    //make a bool to check if the execution is over
    bool *d_over;
    cudaMalloc((void**) &d_over, sizeof(bool));

    #ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_total_start, &tv);
    h2d_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
    #endif

    printf("Copied Everything to GPU memory\n");

    // setup execution parameters
    dim3  grid(num_of_blocks, 1, 1);
    dim3  threads(num_of_threads_per_block, 1, 1);

    int k=0;
    printf("Start traversing the tree\n");
    bool stop;
    do
    {
        stop=false;
        #ifdef  TIMING
        gettimeofday(&tv_h2d_start, NULL);
        #endif
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
        #ifdef  TIMING
        gettimeofday(&tv_h2d_end, NULL);
        tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
        h2d_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
        #endif

        Kernel<<< grid, threads, 0 >>>(d_graph_nodes, d_graph_edges, d_graph_mask, 
                                      d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);

        Kernel2<<< grid, threads, 0 >>>(d_graph_mask, d_updating_graph_mask, 
                                       d_graph_visited, d_over, no_of_nodes);

        #ifdef  TIMING
        cudaDeviceSynchronize();
        gettimeofday(&tv_kernel_end, NULL);
        tvsub(&tv_kernel_end, &tv_h2d_end, &tv);
        kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
        #endif

        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        #ifdef  TIMING
        gettimeofday(&tv_d2h_end, NULL);
        tvsub(&tv_d2h_end, &tv_kernel_end, &tv);
        d2h_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
        #endif

        k++;
    }
    while(stop);

    printf("Kernel Executed %d times\n",k);

    // copy result from device to host
    #ifdef  TIMING
    gettimeofday(&tv_d2h_start, NULL);
    #endif
    cudaMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
    #ifdef  TIMING
    gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
    d2h_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
    #endif

    // Compare CPU and GPU results
    printf("Comparing CPU and GPU results\n");
    bool results_match = true;
    for(int i = 0; i < no_of_nodes; i++) {
        if(h_cost[i] != h_cost_cpu[i]) {
            printf("Mismatch at node %d: GPU cost = %d, CPU cost = %d\n", 
                   i, h_cost[i], h_cost_cpu[i]);
            results_match = false;
        }
    }
    
    // ASCII art output
    if(results_match) {
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
    } else {
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
    }

    // Store both results
    FILE *fpo = fopen("result_gpu.txt","w");
    for(int i=0;i<no_of_nodes;i++)
        fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
    fclose(fpo);
    
    FILE *fpo_cpu = fopen("result_cpu.txt","w");
    for(int i=0;i<no_of_nodes;i++)
        fprintf(fpo_cpu,"%d) cost:%d\n",i,h_cost_cpu[i]);
    fclose(fpo_cpu);
    
    printf("Results stored in result_gpu.txt and result_cpu.txt\n");

    // cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_graph_visited_cpu);
    free(h_cost);
    free(h_cost_cpu);

    #ifdef  TIMING
    gettimeofday(&tv_close_start, NULL);
    #endif
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
    cudaFree(d_over);

    #ifdef  TIMING
    gettimeofday(&tv_close_end, NULL);
    tvsub(&tv_close_end, &tv_close_start, &tv);
    close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
    tvsub(&tv_close_end, &tv_total_start, &tv);
    total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    printf("Init: %f\n", init_time);
    printf("MemAlloc: %f\n", mem_alloc_time);
    printf("HtoD: %f\n", h2d_time);
    printf("Exec: %f\n", kernel_time);
    printf("DtoH: %f\n", d2h_time);
    printf("Close: %f\n", close_time);
    printf("Total: %f\n", total_time);
    #endif
}
