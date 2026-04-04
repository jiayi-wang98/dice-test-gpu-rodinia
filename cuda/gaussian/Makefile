include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

ASAN ?= 0
ASAN_NVCC_FLAGS =
ifeq ($(ASAN),1)
	ASAN_NVCC_FLAGS = -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"
endif

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared $(ASAN_NVCC_FLAGS)

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

clean:
	rm gaussian
