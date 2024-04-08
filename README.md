# Pulscan

Pulscan is a high-performance pulsar searching tool that can run on CPUs as well as GPUs for accelerated computation. This tool is designed to analyze astronomical data for pulsar search, employing various computational optimizations for efficiency.

## Prerequisites

- GCC compiler for C code compilation
- NVIDIA CUDA Toolkit for compiling and running GPU and hybrid codes
- OpenMP for parallel computing on the CPU

## Installation

First, clone the Pulscan repository to your local machine:

```bash
git clone https://github.com/jack-white1/pulscan
cd pulscan
```

### For CPU Version

To compile the CPU version of Pulscan, follow these steps:

1. Compile the local CDF library:

   ```bash
   gcc -c localcdflib.c -o localcdflib.o -lm -Ofast
   ```

2. Compile the Pulscan CPU version:

   ```bash
   gcc pulscan.c localcdflib.o -o pulscan -lm -fopenmp -Ofast
   ```

### For Hybrid (CPU/GPU) Version

For the hybrid version that utilizes both CPU and GPU resources:

1. Compile the local CDF library (same as CPU version):

   ```bash
   gcc -c localcdflib.c -o localcdflib.o -lm -Ofast
   ```

2. Compile the Pulscan hybrid version:

   ```bash
   nvcc pulscan_hybrid.cu localcdflib.o -o pulscan_hybrid -lm -Xcompiler "-fopenmp -Ofast" --use_fast_math
   ```

### For GPU Version

To compile the GPU-only version of Pulscan:

```bash
nvcc pulscan_gpu.cu -o pulscan_gpu -lm -Xcompiler "-fopenmp -Ofast" --use_fast_math
```
```

Feel free to adjust the content to better suit your project's needs or preferences.