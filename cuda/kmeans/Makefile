include ../../common/make.config

ASAN_FLAGS := -fsanitize=address -fno-omit-frame-pointer -lstdc++
# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 $(ASAN_FLAGS)
# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0 -cudart shared -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"
else
	NVCC_FLAGS += -O2 -cudart shared -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu 
endif

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

kmeans.o:
	$(CC) $(CC_FLAGS) kmeans.c kmeans.h -c

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo