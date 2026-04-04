include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

ASAN ?= 0
ASAN_NVCC_FLAGS =
ifeq ($(ASAN),1)
	ASAN_NVCC_FLAGS = -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"
endif

SRC = bfs.cu

EXE = bfs.out

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -I../util -L$(CUDA_LIB_DIR) -DTIMING -cudart shared $(ASAN_NVCC_FLAGS)

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
