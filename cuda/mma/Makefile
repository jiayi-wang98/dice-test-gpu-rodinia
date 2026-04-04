include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : wmma_demo wmma_1024

clean :
	rm -rf *.o wmma_demo

wmma_demo : wmma_demo.cu
	$(CC) -arch=sm_70 -ptx wmma_demo.cu -o wmma_demo.ptx
	$(CC) -cuda wmma_demo.cu -cudart shared -arch=sm_70
	$(CC) -o wmma_demo wmma_demo.cu -cudart shared -arch=sm_70
	
wmma_1024 : wmma_1024.cu
	$(CC) -arch=sm_70 -ptx wmma_1024.cu -o wmma_1024.ptx
	$(CC) -cuda wmma_1024.cu -cudart shared -arch=sm_70
	$(CC) -o wmma_1024 wmma_1024.cu -cudart shared -arch=sm_70
