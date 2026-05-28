include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = softmax_mr.cu
EXE = softmax_mr
ROWS_PER_CTA ?= 8

release:
	$(CC) -DROWS_PER_CTA=$(ROWS_PER_CTA) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

debug:
	$(CC) -DROWS_PER_CTA=$(ROWS_PER_CTA) -g -G $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -cudart shared

clean:
	rm -f $(EXE)
