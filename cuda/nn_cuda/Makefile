include ../../common/make.config

SHELL := /bin/bash
LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc
DICE_GPGPU_SIM_ROOT ?= $(abspath ../../../dice_gpgpu-sim)
DICE_SETUP_ENV ?= $(DICE_GPGPU_SIM_ROOT)/setup_environment
all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	. $(DICE_SETUP_ENV) debug && \
	$(CC) -cuda nn_cuda.cu -cudart shared
	$(CC) -o nn nn_cuda.cu -cudart shared

clang: $(SRC)
	clang++ nn_cuda.cu -o nn -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
