include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = softmax_v2.cu
EXE = softmax_v2
K ?= 4

release:
	$(CC) -DK=$(K) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

debug:
	$(CC) -DK=$(K) -g -G $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

clean:
	rm -f $(EXE)
