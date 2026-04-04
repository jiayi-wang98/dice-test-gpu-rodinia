include ../../common/make.config


# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_INC = -I$(CUDA_DIR)/include

# 'make dbg=1' enables NVCC debugging
NVCC_FLAGS = $(NVCC_INC)
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0 -cudart shared
else
	NVCC_FLAGS += -O2 -cudart shared
endif

# 'make emu=1' compiles the CUDA kernels for emulation (deprecated on modern CUDA)
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

# -----------------------------
# PTX generation
# -----------------------------
# Usage:
#   make ptx
#   make ptx PTX_ARCH=sm_80
#   make ptx PTX_ARCH=sm_75
PTX_ARCH ?= sm_75

# Disable loop unrolling in ptxas (best-effort)
# (You can also add -O0 if you want: PTX_FLAGS := -ptx -O0 ...)
PTX_FLAGS := -ptx -arch=$(PTX_ARCH) -Xptxas --disable-optimizer-unroll

# For PTX, avoid host sanitizer flags; only keep include paths
PTX_NVCC_FLAGS := $(NVCC_INC)

# -----------------------------
# Targets
# -----------------------------
all: kmeans

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o \
		-o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

# Build PTX
ptx: kmeans_cuda.ptx

kmeans_cuda.ptx: kmeans_cuda.cu
	$(NVCC) $(PTX_NVCC_FLAGS) $(PTX_FLAGS) $< -o $@

kmeans.o: kmeans.c kmeans.h
	$(CC) $(CC_FLAGS) -c kmeans.c

%.o: %.[ch]
	$(CC) $(CC_FLAGS) -c $<

kmeans_cuda.o: kmeans_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo kmeans_cuda.ptx