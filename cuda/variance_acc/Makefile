include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = variance_acc.cu
EXE = variance_acc

release:
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

debug:
	$(CC) -g -G $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

clean:
	rm -f $(EXE)
