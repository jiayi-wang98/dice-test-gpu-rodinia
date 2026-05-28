include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = softmax_warp.cu
EXE = softmax_warp

release:
	$(CC) -Xptxas -w $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared -arch=sm_52

debug:
	$(CC) -g -G $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared -arch=sm_52

clean:
	rm -f $(EXE)
