include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	. /data/jwang710/dice_gpgpu-sim/setup_environment debug && \
	$(CC) -cuda nn_cuda.cu -cudart shared -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"
	$(CC) -o nn nn_cuda.cu -cudart shared -Xcompiler "-fsanitize=address -fno-omit-frame-pointer"

clang: $(SRC)
	clang++ nn_cuda.cu -o nn -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
