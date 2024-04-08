# Pulscan

Pulscan is a high-performance pulsar searching tool that can run on CPUs as well as GPUs for accelerated computation. This tool performs an acceleration search for the signature of binary pulsars using boxcar filters on FFT spectra. 

The input data is expected to be in .fft format.

A .fft file is a binary file consisting of an even number of FP32 floats, where each pair represents the real and complex component of a single bin of an FFT spectrum. The first pair of floats represents the DC, or zero frequency component of the signal. The final pair of numbers should be the frequency component corresponding to sampling_frequency/2.

## Prerequisites

- GCC compiler for C code compilation
- OpenMP for parallel computing on the CPU
- NVIDIA CUDA Toolkit for compiling and running GPU and hybrid codes


## Installation

First, clone the Pulscan repository to your local machine:

```bash
git clone https://github.com/jack-white1/pulscan
cd pulscan
```

### For GPU Version

To compile the GPU-only version of Pulscan:

```bash
nvcc pulscan_gpu.cu -o pulscan_gpu -lm -Xcompiler "-fopenmp -Ofast" --use_fast_math
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