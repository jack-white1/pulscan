# Makefile for Pulscan

# Compiler and flags
GCC = gcc
NVCC = nvcc
CFLAGS = -lm -Ofast
OMP_FLAGS = -fopenmp
NVCC_GPU_FLAGS = -lm -Xcompiler "-fopenmp -Ofast" --use_fast_math -lcufft -lineinfo -arch=sm_90
NVCC_HYBRID_FLAGS = -lm -Xcompiler "-fopenmp -Ofast" --use_fast_math -lcufft

# Default target
all: gpu cpu

# Compile CPU version
cpu:
	$(GCC) -c localcdflib.c -o localcdflib.o $(CFLAGS)
	$(GCC) pulscan.c localcdflib.o -o pulscan $(CFLAGS) $(OMP_FLAGS)

# Compile GPU version
gpu:
	$(GCC) -c localcdflib.c -o localcdflib.o $(CFLAGS)
	$(NVCC) pulscan_hybrid.cu localcdflib.o -o pulscan_hybrid $(NVCC_HYBRID_FLAGS)
	$(NVCC) pulscan_gpu.cu localcdflib.o -o pulscan_gpu $(NVCC_GPU_FLAGS)
	$(NVCC) pulscan_gpu_bfloat16.cu localcdflib.o -o pulscan_gpu_bf16 $(NVCC_GPU_FLAGS)

# Clean up
clean:
	rm -f localcdflib.o pulscan pulscan_hybrid pulscan_gpu

# Phony targets
.PHONY: all clean gpu cpu
