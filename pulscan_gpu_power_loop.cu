#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

struct candidate{
    float power;
    float logp;
    int r;
    int z;
    int numharm;
};

typedef struct {
    float power;
    int index;
} power_index_struct;


// comparison function for qsort
int compareCandidatesByLogp(const void* a, const void* b){
    candidate* candidateA = (candidate*)a;
    candidate* candidateB = (candidate*)b;
    if (candidateA->logp > candidateB->logp){
        return 1;
    } else if (candidateA->logp < candidateB->logp){
        return -1;
    } else {
        return 0;
    }
}

double __device__ power_to_logp(float chi2, float dof){
    double double_dof = (double) dof;
    double double_chi2 = (double) chi2;
    // Use boundary condition
    if (dof >= chi2 * 1.05){
        return 0.0;
    } else {
        double x = 1500 * double_dof / double_chi2;
        // Updated polynomial equation
        double f_x = (-4.460405902717228e-46 * pow(x, 16) + 9.492786384945832e-42 * pow(x, 15) - 
               9.147045144529116e-38 * pow(x, 14) + 5.281085384219971e-34 * pow(x, 13) - 
               2.0376166670276118e-30 * pow(x, 12) + 5.548033164083744e-27 * pow(x, 11) - 
               1.0973877021703706e-23 * pow(x, 10) + 1.5991806841151474e-20 * pow(x, 9) - 
               1.7231488066853853e-17 * pow(x, 8) + 1.3660070957914896e-14 * pow(x, 7) - 
               7.861795249869729e-12 * pow(x, 6) + 3.2136336591718867e-09 * pow(x, 5) - 
               9.046641813341226e-07 * pow(x, 4) + 0.00016945948004599545 * pow(x, 3) - 
               0.0214942314851717 * pow(x, 2) + 2.951595476316614 * x - 
               755.240918031251);
        double logp = chi2 * f_x / 1500;
        return logp;
    }
}

__global__ void wakeGPUKernel(){
    // This kernel does nothing, it is used to wake up the GPU
    // so that the first kernel run is not slow
    float a = 1.0;
    float b = 2.0;
    float c = a + b;
}

__global__ void separateRealAndImaginaryComponents(float2* rawDataDevice, float* realData, float* imaginaryData, long numComplexFloats){
    long globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numComplexFloats){
        float2 currentValue = rawDataDevice[globalThreadIndex];
        realData[globalThreadIndex] = currentValue.x;
        imaginaryData[globalThreadIndex] = currentValue.y;
    }
}

__global__ void medianOfMediansNormalisation(float* globalArray) {
    // HARDCODED FOR PERFORMANCE
    // USE medianOfMediansNormalisationAnyBlockSize() FOR GENERAL USE

    // Each thread loads 4 elements from global memory to shared memory
    // then calculates the median of these 4 elements, recursively reducing the array down to 
    //      a single median of medians value
    // then subtracts the median of medians from each element
    // then takes the absolute value of each element
    // then calculates the median of these absolute values
    // then multiplies this new median (aka median absolute deviation) by 1.4826
    // then subtracts the median from each original element and divides by the new median absolute deviation

    // Assumes blockDim.x = 1024
    // TODO: make this work for any blockDim.x
    __shared__ float medianArray[4096];
    __shared__ float madArray[4096];
    __shared__ float normalisedArray[4096];

    //int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;
    int globalArrayIndex = blockDim.x*blockIdx.x*4+threadIdx.x;

    float median;
    float mad;

    medianArray[localThreadIndex] = globalArray[globalArrayIndex];
    medianArray[localThreadIndex + 1024] = globalArray[globalArrayIndex + 1024];
    medianArray[localThreadIndex + 2048] = globalArray[globalArrayIndex + 2048];
    medianArray[localThreadIndex + 3072] = globalArray[globalArrayIndex + 3072];

    madArray[localThreadIndex] = medianArray[localThreadIndex];
    madArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    madArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    madArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    normalisedArray[localThreadIndex] = medianArray[localThreadIndex];
    normalisedArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    normalisedArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    normalisedArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    __syncthreads();

    float a,b,c,d,min,max;
  
    a = medianArray[localThreadIndex];
    b = medianArray[localThreadIndex+1024];
    c = medianArray[localThreadIndex+2048];
    d = medianArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();
    if(localThreadIndex < 256){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+256];
        c = medianArray[localThreadIndex+512];
        d = medianArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+64];
        c = medianArray[localThreadIndex+128];
        d = medianArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+16];
        c = medianArray[localThreadIndex+32];
        d = medianArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+4];
        c = medianArray[localThreadIndex+8];
        d = medianArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+1];
        c = medianArray[localThreadIndex+2];
        d = medianArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }

    __syncthreads();

    median = medianArray[0];
    __syncthreads();

    //if (localThreadIndex == 0){
    //    printf("madArray[0]: %f, medianArray[0]: %f\n", madArray[0], medianArray[0]);
    //}

    madArray[localThreadIndex] = fabsf(madArray[localThreadIndex] - median);
    madArray[localThreadIndex + 1024] = fabsf(madArray[localThreadIndex + 1024] - median);
    madArray[localThreadIndex + 2048] = fabsf(madArray[localThreadIndex + 2048] - median);
    madArray[localThreadIndex + 3072] = fabsf(madArray[localThreadIndex + 3072] - median);

    //if (localThreadIndex == 0){
    //    printf("fabsf(madArray[0]): %f, medianArray[0]: %f\n", madArray[0], medianArray[0]);
    //}
    __syncthreads();

    a = madArray[localThreadIndex];
    b = madArray[localThreadIndex+1024];
    c = madArray[localThreadIndex+2048];
    d = madArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();

    if(localThreadIndex < 512){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+512];
        c = madArray[localThreadIndex+1024];
        d = madArray[localThreadIndex+1536];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 256){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+256];
        c = madArray[localThreadIndex+512];
        d = madArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+64];
        c = madArray[localThreadIndex+128];
        d = madArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+16];
        c = madArray[localThreadIndex+32];
        d = madArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+4];
        c = madArray[localThreadIndex+8];
        d = madArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+1];
        c = madArray[localThreadIndex+2];
        d = madArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5*1.4826;
        //printf("a=%f,b=%f,c=%f,d=%f,min=%f,max=%f,1/mad=%f,mad=%.16f\n",a,b,c,d,min,max,madArray[localThreadIndex],1/madArray[localThreadIndex]);
        
    }
    __syncthreads();
    mad =  madArray[0];
    __syncthreads();


    normalisedArray[localThreadIndex] = (normalisedArray[localThreadIndex] - median) / mad;
    normalisedArray[localThreadIndex + 1024] = (normalisedArray[localThreadIndex + 1024] - median) / mad;
    normalisedArray[localThreadIndex + 2048] = (normalisedArray[localThreadIndex + 2048] - median) / mad;
    normalisedArray[localThreadIndex + 3072] = (normalisedArray[localThreadIndex + 3072] - median) / mad;

    __syncthreads();

    globalArray[globalArrayIndex] = normalisedArray[localThreadIndex];
    globalArray[globalArrayIndex + 1024] = normalisedArray[localThreadIndex + 1024];
    globalArray[globalArrayIndex + 2048] = normalisedArray[localThreadIndex + 2048];
    globalArray[globalArrayIndex + 3072] = normalisedArray[localThreadIndex + 3072];

    //if (localThreadIndex == 0){
    //    printf("%f,%f,%f,%f\n",globalArray[globalThreadIndex],globalArray[globalThreadIndex + 1024],globalArray[globalThreadIndex + 2048],globalArray[globalThreadIndex + 3072]);
    //}

    //if (localThreadIndex == 0){
    //    printf("Median: %f, MAD: %f\n", median, mad);
    //}
}
__global__ void medianOfMediansNormalisationOLD(float* globalArray) {
    // HARDCODED FOR PERFORMANCE
    // USE medianOfMediansNormalisationAnyBlockSize() FOR GENERAL USE

    // Each thread loads 4 elements from global memory to shared memory
    // then calculates the median of these 4 elements, recursively reducing the array down to 
    //      a single median of medians value
    // then subtracts the median of medians from each element
    // then takes the absolute value of each element
    // then calculates the median of these absolute values
    // then multiplies this new median (aka median absolute deviation) by 1.4826
    // then subtracts the median from each original element and divides by the new median absolute deviation

    // Assumes blockDim.x = 1024
    // TODO: make this work for any blockDim.x
    __shared__ float medianArray[4096];
    __shared__ float madArray[4096];
    __shared__ float normalisedArray[4096];

    //int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;
    int globalArrayIndex = blockDim.x*blockIdx.x*4+threadIdx.x;

    float median;
    float mad;

    medianArray[localThreadIndex] = globalArray[globalArrayIndex];
    medianArray[localThreadIndex + 1024] = globalArray[globalArrayIndex + 1024];
    medianArray[localThreadIndex + 2048] = globalArray[globalArrayIndex + 2048];
    medianArray[localThreadIndex + 3072] = globalArray[globalArrayIndex + 3072];

    madArray[localThreadIndex] = medianArray[localThreadIndex];
    madArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    madArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    madArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    normalisedArray[localThreadIndex] = medianArray[localThreadIndex];
    normalisedArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    normalisedArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    normalisedArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    __syncthreads();

    float a,b,c,d,min,max;
  
    a = medianArray[localThreadIndex];
    b = medianArray[localThreadIndex+1024];
    c = medianArray[localThreadIndex+2048];
    d = medianArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();

    if(localThreadIndex < 512){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+512];
        c = medianArray[localThreadIndex+1024];
        d = medianArray[localThreadIndex+1536];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 256){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+256];
        c = medianArray[localThreadIndex+512];
        d = medianArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+64];
        c = medianArray[localThreadIndex+128];
        d = medianArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+16];
        c = medianArray[localThreadIndex+32];
        d = medianArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+4];
        c = medianArray[localThreadIndex+8];
        d = medianArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+1];
        c = medianArray[localThreadIndex+2];
        d = medianArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }

    __syncthreads();

    median = medianArray[0];
    __syncthreads();

    //if (localThreadIndex == 0){
    //    printf("madArray[0]: %f, medianArray[0]: %f\n", madArray[0], medianArray[0]);
    //}

    madArray[localThreadIndex] = fabsf(madArray[localThreadIndex] - median);
    madArray[localThreadIndex + 1024] = fabsf(madArray[localThreadIndex + 1024] - median);
    madArray[localThreadIndex + 2048] = fabsf(madArray[localThreadIndex + 2048] - median);
    madArray[localThreadIndex + 3072] = fabsf(madArray[localThreadIndex + 3072] - median);

    //if (localThreadIndex == 0){
    //    printf("fabsf(madArray[0]): %f, medianArray[0]: %f\n", madArray[0], medianArray[0]);
    //}
    __syncthreads();

    a = madArray[localThreadIndex];
    b = madArray[localThreadIndex+1024];
    c = madArray[localThreadIndex+2048];
    d = madArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();

    if(localThreadIndex < 512){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+512];
        c = madArray[localThreadIndex+1024];
        d = madArray[localThreadIndex+1536];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 256){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+256];
        c = madArray[localThreadIndex+512];
        d = madArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+64];
        c = madArray[localThreadIndex+128];
        d = madArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+16];
        c = madArray[localThreadIndex+32];
        d = madArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+4];
        c = madArray[localThreadIndex+8];
        d = madArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+1];
        c = madArray[localThreadIndex+2];
        d = madArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5*1.4826;
        //printf("a=%f,b=%f,c=%f,d=%f,min=%f,max=%f,1/mad=%f,mad=%.16f\n",a,b,c,d,min,max,madArray[localThreadIndex],1/madArray[localThreadIndex]);
        
    }
    __syncthreads();
    mad =  madArray[0];
    __syncthreads();


    normalisedArray[localThreadIndex] = (normalisedArray[localThreadIndex] - median) / mad;
    normalisedArray[localThreadIndex + 1024] = (normalisedArray[localThreadIndex + 1024] - median) / mad;
    normalisedArray[localThreadIndex + 2048] = (normalisedArray[localThreadIndex + 2048] - median) / mad;
    normalisedArray[localThreadIndex + 3072] = (normalisedArray[localThreadIndex + 3072] - median) / mad;

    __syncthreads();

    globalArray[globalArrayIndex] = normalisedArray[localThreadIndex];
    globalArray[globalArrayIndex + 1024] = normalisedArray[localThreadIndex + 1024];
    globalArray[globalArrayIndex + 2048] = normalisedArray[localThreadIndex + 2048];
    globalArray[globalArrayIndex + 3072] = normalisedArray[localThreadIndex + 3072];

    //if (localThreadIndex == 0){
    //    printf("%f,%f,%f,%f\n",globalArray[globalThreadIndex],globalArray[globalThreadIndex + 1024],globalArray[globalThreadIndex + 2048],globalArray[globalThreadIndex + 3072]);
    //}

    //if (localThreadIndex == 0){
    //    printf("Median: %f, MAD: %f\n", median, mad);
    //}
}

/*
__global__ void medianOfMediansNormalisationAnyBlockSize(float* globalArray) {
    extern __shared__ float sharedMemory[];
    // Each thread loads 4 elements from global memory to shared memory
    __shared__ float* medianArray = &sharedMemory[0];
    __shared__ float* madArray = &sharedMemory[blockDim.x];
    __shared__ float* normalisedArray = &sharedMemory[2*blockDim.x];

    int localThreadIndex = threadIdx.x;
    int globalArrayIndex = blockDim.x*blockIdx.x*4+threadIdx.x;

    float a,b,c,d,min,max,median,mad;

    medianArray[localThreadIndex] = globalArray[globalArrayIndex];
    medianArray[localThreadIndex + blockDim.x] = globalArray[globalArrayIndex + blockDim.x];
    medianArray[localThreadIndex + 2*blockDim.x] = globalArray[globalArrayIndex + 2*blockDim.x];
    medianArray[localThreadIndex + 3*blockDim.x] = globalArray[globalArrayIndex + 3*blockDim.x];

    madArray[localThreadIndex] = medianArray[localThreadIndex];
    madArray[localThreadIndex + blockDim.x] = medianArray[localThreadIndex + blockDim.x];
    madArray[localThreadIndex + 2*blockDim.x] = medianArray[localThreadIndex + 2*blockDim.x];
    madArray[localThreadIndex + 3*blockDim.x] = medianArray[localThreadIndex + 3*blockDim.x];

    normalisedArray[localThreadIndex] = medianArray[localThreadIndex];
    normalisedArray[localThreadIndex + blockDim.x] = medianArray[localThreadIndex + blockDim.x];
    normalisedArray[localThreadIndex + 2*blockDim.x] = medianArray[localThreadIndex + 2*blockDim.x];
    normalisedArray[localThreadIndex + 3*blockDim.x] = medianArray[localThreadIndex + 3*blockDim.x];

    __syncthreads();

    for (int stride = blockDim.x; stride > 0; stride >>= 1){
        if(localThreadIndex < stride){
            a = medianArray[localThreadIndex];
            b = medianArray[localThreadIndex+stride];
            c = medianArray[localThreadIndex+2*stride];
            d = medianArray[localThreadIndex+3*stride];
            min = fminf(fminf(fminf(a,b),c),d);
            max = fmaxf(fmaxf(fmaxf(a,b),c),d);
            medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        }
        __syncthreads();
    }
  

    median = medianArray[0];
    __syncthreads();

    madArray[localThreadIndex] = fabsf(madArray[localThreadIndex] - median);
    madArray[localThreadIndex + blockDim.x] = fabsf(madArray[localThreadIndex + blockDim.x] - median);
    madArray[localThreadIndex + 2*blockDim.x] = fabsf(madArray[localThreadIndex + 2*blockDim.x] - median);
    madArray[localThreadIndex + 3*blockDim.x] = fabsf(madArray[localThreadIndex + 3*blockDim.x] - median);

    __syncthreads();

    for (int stride = blockDim.x; stride > 0; stride >>= 1){
        if(localThreadIndex < stride){
            a = madArray[localThreadIndex];
            b = madArray[localThreadIndex+stride];
            c = madArray[localThreadIndex+2*stride];
            d = madArray[localThreadIndex+3*stride];
            min = fminf(fminf(fminf(a,b),c),d);
            max = fmaxf(fmaxf(fmaxf(a,b),c),d);
            madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        }
        __syncthreads();
    }

    mad =  madArray[0];
    __syncthreads();

    normalisedArray[localThreadIndex] = (normalisedArray[localThreadIndex] - median) / mad;
    normalisedArray[localThreadIndex + blockDim.x] = (normalisedArray[localThreadIndex + blockDim.x] - median) / mad;
    normalisedArray[localThreadIndex + 2*blockDim.x] = (normalisedArray[localThreadIndex + 2*blockDim.x] - median) / mad;
    normalisedArray[localThreadIndex + 3*blockDim.x] = (normalisedArray[localThreadIndex + 3*blockDim.x] - median) / mad;

    __syncthreads();

    globalArray[globalArrayIndex] = normalisedArray[localThreadIndex];
    globalArray[globalArrayIndex + blockDim.x] = normalisedArray[localThreadIndex + blockDim.x];
    globalArray[globalArrayIndex + 2*blockDim.x] = normalisedArray[localThreadIndex + 2*blockDim.x];
    globalArray[globalArrayIndex + 3*blockDim.x] = normalisedArray[localThreadIndex + 3*blockDim.x];
}
*/

__global__ void magnitudeSquared(float* realData, float* imaginaryData, float* magnitudeSquaredArray, long numFloats){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numFloats){
        float real = realData[globalThreadIndex];
        float imaginary = imaginaryData[globalThreadIndex];
        magnitudeSquaredArray[globalThreadIndex] = real*real + imaginary*imaginary;
    }
}


// takes a 1D array like this:
// magnitudeSquaredArray:   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// and adds these elements together, effectively performing a harmonic sum
// decimatedArray2:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray3:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray4:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,x,x,x,x,0,0,0,0,0,0,0,0,0]
//                                     |<--------->|<--------->|<--------->|
//                                        equal spacing between harmonics

__global__ void decimateHarmonics(float* magnitudeSquaredArray, float* decimatedArray2, float* decimatedArray3, float* decimatedArray4, long numMagnitudes){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;

    float fundamental;
    float harmonic1a, harmonic1b;
    float harmonic2a, harmonic2b, harmonic2c;
    float harmonic3a, harmonic3b, harmonic3c, harmonic3d;

    if (globalThreadIndex*2+1 < numMagnitudes){
        fundamental = magnitudeSquaredArray[globalThreadIndex];
        harmonic1a = magnitudeSquaredArray[globalThreadIndex*2];
        harmonic1b = magnitudeSquaredArray[globalThreadIndex*2+1];
        decimatedArray2[globalThreadIndex] = fundamental+harmonic1a+harmonic1b;
    }

    if (globalThreadIndex*3+2 < numMagnitudes){
        harmonic2a = magnitudeSquaredArray[globalThreadIndex*3];
        harmonic2b = magnitudeSquaredArray[globalThreadIndex*3+1];
        harmonic2c = magnitudeSquaredArray[globalThreadIndex*3+2];
        decimatedArray3[globalThreadIndex] = fundamental+harmonic1a+harmonic1b
                                                +harmonic2a+harmonic2b+harmonic2c;
    }

    if (globalThreadIndex*4+3 < numMagnitudes){
        harmonic3a = magnitudeSquaredArray[globalThreadIndex*4];
        harmonic3b = magnitudeSquaredArray[globalThreadIndex*4+1];
        harmonic3c = magnitudeSquaredArray[globalThreadIndex*4+2];
        harmonic3d = magnitudeSquaredArray[globalThreadIndex*4+3];
        decimatedArray4[globalThreadIndex] = fundamental+harmonic1a+harmonic1b
                                                +harmonic2a+harmonic2b+harmonic2c
                                                +harmonic3a+harmonic3b+harmonic3c+harmonic3d;
    }

}

// logarithmic zstep, zmax = 256, numThreads = 256
__global__ void boxcarFilterArrayROLLED(float* magnitudeSquaredArray, candidate* globalCandidateArray, int numharm, long numFloats, int numCandidatesPerBlock){
    __shared__ float lookupArray[512];
    __shared__ float sumArray[256];
    __shared__ power_index_struct searchArray[256];
    __shared__ candidate localCandidateArray[16]; //oversized, has to be greater than numCandidatesPerBlock

    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;

    lookupArray[localThreadIndex] = magnitudeSquaredArray[globalThreadIndex];
    lookupArray[localThreadIndex + 256] = magnitudeSquaredArray[globalThreadIndex + 256];

    __syncthreads();

    // initialise the sum array
    sumArray[localThreadIndex] = 0.0f;
    __syncthreads();
    // begin boxcar filtering
    int targetZ = 0;
    int outputCounter = 0;

    for (int z = 0; z <= 256; z+=1){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
        if (z == targetZ){
            searchArray[localThreadIndex].power = sumArray[localThreadIndex];
            searchArray[localThreadIndex].index = globalThreadIndex;
            for (int stride = blockDim.x / 2; stride>0; stride /= 2){
                if (localThreadIndex < stride){
                    if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power){
                        searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
                    }
                }
                __syncthreads();
            }
            if (localThreadIndex == 0){
                localCandidateArray[outputCounter].power = searchArray[0].power;
                localCandidateArray[outputCounter].r =searchArray[0].index;
                localCandidateArray[outputCounter].z = z;
                localCandidateArray[outputCounter].logp = 0.0f;
                localCandidateArray[outputCounter].numharm = numharm;
            }
            outputCounter+=1;
            if (targetZ == 0){
                targetZ = 1;
            } else {
                targetZ *= 2;
            }
        }
        __syncthreads();
    }

    __syncthreads();

    if (localThreadIndex < numCandidatesPerBlock){
        globalCandidateArray[blockIdx.x*numCandidatesPerBlock+localThreadIndex] = localCandidateArray[localThreadIndex];
    }
}

__global__ void boxcarFilterArray(float* magnitudeSquaredArray, candidate* globalCandidateArray, int numharm, long numFloats, int numCandidatesPerBlock){
    __shared__ float lookupArray[512];
    __shared__ float sumArray[256];
    __shared__ power_index_struct searchArray[256];
    __shared__ candidate localCandidateArray[16]; // oversized, has to be greater than numCandidatesPerBlock

    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;

    lookupArray[localThreadIndex] = magnitudeSquaredArray[globalThreadIndex];
    lookupArray[localThreadIndex + 256] = magnitudeSquaredArray[globalThreadIndex + 256];

    __syncthreads();

    // initialize the sum array
    sumArray[localThreadIndex] = 0.0f;
    __syncthreads();
    
    // begin boxcar filtering
    int outputCounter = 0;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex];
    // search at z = 1
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 1;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 1];
    // search at z = 2
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 2;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 2];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 3];
    // search at z = 4
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 4;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 4];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 5];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 6];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 7];
    // search at z = 8
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 8;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 8];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 9];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 10];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 11];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 12];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 13];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 14];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 15];
    // search at z = 16
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 16;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 16];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 17];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 18];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 19];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 20];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 21];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 22];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 23];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 24];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 25];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 26];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 27];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 28];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 29];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 30];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 31];
    // search at z = 32
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 32;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 32];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 33];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 34];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 35];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 36];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 37];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 38];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 39];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 40];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 41];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 42];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 43];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 44];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 45];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 46];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 47];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 48];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 49];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 50];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 51];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 52];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 53];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 54];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 55];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 56];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 57];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 58];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 59];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 60];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 61];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 62];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 63];
    // search at z = 64
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 64;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 64];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 65];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 66];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 67];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 68];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 69];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 70];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 71];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 72];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 73];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 74];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 75];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 76];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 77];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 78];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 79];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 80];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 81];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 82];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 83];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 84];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 85];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 86];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 87];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 88];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 89];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 90];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 91];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 92];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 93];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 94];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 95];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 96];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 97];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 98];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 99];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 100];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 101];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 102];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 103];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 104];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 105];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 106];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 107];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 108];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 109];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 110];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 111];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 112];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 113];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 114];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 115];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 116];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 117];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 118];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 119];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 120];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 121];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 122];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 123];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 124];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 125];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 126];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 127];
    // search at z = 128
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 128;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 128];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 129];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 130];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 131];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 132];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 133];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 134];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 135];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 136];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 137];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 138];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 139];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 140];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 141];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 142];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 143];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 144];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 145];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 146];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 147];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 148];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 149];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 150];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 151];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 152];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 153];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 154];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 155];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 156];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 157];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 158];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 159];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 160];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 161];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 162];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 163];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 164];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 165];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 166];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 167];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 168];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 169];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 170];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 171];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 172];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 173];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 174];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 175];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 176];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 177];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 178];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 179];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 180];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 181];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 182];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 183];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 184];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 185];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 186];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 187];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 188];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 189];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 190];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 191];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 192];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 193];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 194];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 195];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 196];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 197];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 198];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 199];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 200];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 201];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 202];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 203];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 204];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 205];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 206];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 207];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 208];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 209];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 210];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 211];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 212];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 213];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 214];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 215];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 216];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 217];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 218];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 219];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 220];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 221];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 222];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 223];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 224];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 225];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 226];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 227];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 228];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 229];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 230];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 231];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 232];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 233];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 234];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 235];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 236];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 237];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 238];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 239];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 240];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 241];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 242];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 243];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 244];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 245];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 246];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 247];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 248];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 249];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 250];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 251];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 252];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 253];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 254];
    sumArray[localThreadIndex] += lookupArray[localThreadIndex + 255];
    // search at z = 256
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = globalThreadIndex;
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localThreadIndex < stride) {
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power) {
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0) {
        localCandidateArray[outputCounter].power = searchArray[0].power;
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = 256;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
    outputCounter += 1;

    __syncthreads();

    if (localThreadIndex < numCandidatesPerBlock) {
        globalCandidateArray[blockIdx.x * numCandidatesPerBlock + localThreadIndex] = localCandidateArray[localThreadIndex];
    }
}


__global__ void calculateLogp(candidate* globalCandidateArray, long numCandidates, int numSum){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numCandidates){
        double logp = power_to_logp(globalCandidateArray[globalThreadIndex].power,globalCandidateArray[globalThreadIndex].z*numSum*2);
        globalCandidateArray[globalThreadIndex].logp = (float) logp;
    }
}

void copyDeviceArrayToHostAndPrint(float* deviceArray, long numFloats){
    float* hostArray;
    hostArray = (float*)malloc(sizeof(float)*numFloats);
    cudaMemcpy(hostArray, deviceArray, sizeof(float)*numFloats,cudaMemcpyDeviceToHost);
    for (int i = 0; i < numFloats; i++){
        printf("%f\n", hostArray[i]);
    }
    free(hostArray);
}

void copyDeviceArrayToHostAndSaveToFile(float* deviceArray, long numFloats, const char* filename){
    float* hostArray;
    hostArray = (float*)malloc(sizeof(float)*numFloats);
    cudaMemcpy(hostArray, deviceArray, sizeof(float)*numFloats,cudaMemcpyDeviceToHost);
    FILE *f = fopen(filename, "wb");
    // write in csv format, one number per column
    for (int i = 0; i < numFloats; i++){
        fprintf(f, "%f\n", hostArray[i]);
    }
    fclose(f);
    free(hostArray);
}

void copyDeviceCandidateArrayToHostAndPrint(candidate* deviceArray, long numCandidates){
    candidate* hostArray;
    hostArray = (candidate*)malloc(sizeof(candidate)*numCandidates);
    cudaMemcpy(hostArray, deviceArray, sizeof(candidate)*numCandidates,cudaMemcpyDeviceToHost);
    for (int i = 0; i < numCandidates; i++){
        printf("Candidate %d: power: %f, logp: %f, r: %d, z: %d, numharm: %d\n", i, hostArray[i].power, hostArray[i].logp, hostArray[i].r, hostArray[i].z, hostArray[i].numharm);
    }
    free(hostArray);
}

#define RESET   "\033[0m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"

const char* pulscan_frame = 
"    .          .     .     *        .   .   .     .\n"
"         " BOLD "___________      . __" RESET " .  .   *  .   .  .  .     .\n"
"    . *   " BOLD "_____  __ \\__+ __/ /_____________ _____" RESET " .    " FLASHING "*" RESET "  .\n"
"  +    .   " BOLD "___  /_/ / / / / / ___/ ___/ __ `/ __ \\" RESET "     + .\n"
" .          " BOLD "_  ____/ /_/ / (__  ) /__/ /_/ / / / /" RESET " .  *     . \n"
"       .    " BOLD "/_/ *  \\__,_/_/____/\\___/\\__,_/_/ /_/" RESET "    \n"
"    *    +     .     .     . +     .     +   .      *   +\n"

"  J. White, K. Adámek, J. Roy, S. Ransom, W. Armour  2023\n\n";

int main(int argc, char* argv[]){
    int debug = 0;
    printf("%s\n", pulscan_frame);

    // start high resolution timer to measure gpu initialisation time using chrono
    auto start_chrono = std::chrono::high_resolution_clock::now();
    
    cudaDeviceSynchronize();
    wakeGPUKernel<<<1,1>>>();

    auto end_chrono = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    
    printf("GPU initialisation took:                %f ms\n",(float)duration.count());
    
    // start timing
    start_chrono = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        printf("Please provide the input file path as a command line argument.\n");
        return 1;
    }

    const char* filepath = argv[1];

    // Check filepath ends with ".fft"
    if (strlen(filepath) < 5 || strcmp(filepath + strlen(filepath) - 4, ".fft") != 0) {
        printf("Input file must be a .fft file.\n");
        return 1;
    }

    FILE *f = fopen(filepath, "rb");

    // Determine the size of the file in bytes
    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Read the file into CPU memory
    size_t numFloats = filesize / sizeof(float);

    // Cap the filesize at the nearest lower factor of 8192 for compatibility later on
    numFloats = numFloats - (numFloats % 8192);
    float* rawData = (float*) malloc(sizeof(float) * numFloats);
    fread(rawData, sizeof(float), numFloats, f);
    fclose(f);

    // stop timing
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Reading file took:                      %f ms\n", (float)duration.count());

    // start GPU timing
    cudaEvent_t start, stop, overallGPUStart, overallGPUStop;
    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate a suitable array on the GPU, copy raw data across
    float* rawDataDevice;
    cudaMalloc((void**)&rawDataDevice, sizeof(float) * numFloats);
    cudaMemcpy(rawDataDevice, rawData, sizeof(float) * numFloats, cudaMemcpyHostToDevice);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Copying data to GPU took:               %f ms\n", milliseconds);

    int numMagnitudes = numFloats/2;

    // Separate the interleaved real and imaginary parts as the raw data
    // is in the format [[real,imaginary], [real,imaginary]]
    // this is basically a transpose from a 2xN to an Nx2 array
    float* realDataDevice;
    float* imaginaryDataDevice;
    cudaMalloc((void**)&realDataDevice, sizeof(float)*numMagnitudes);
    cudaMalloc((void**)&imaginaryDataDevice, sizeof(float)*numMagnitudes);

    int numThreadsSeparate = 256;
    int numBlocksSeparate = (numMagnitudes + numThreadsSeparate - 1)/ numThreadsSeparate;
    if (debug == 1) {
        printf("Calling separateRealAndImaginaryComponents with %d threads per block and %d blocks\n", numThreadsSeparate, numBlocksSeparate);
    }

    // Normalise the real and imaginary parts

    int numMagnitudesPerThreadNormalise = 4;
    int numThreadsNormalise = 1024; // DO NOT CHANGE THIS, TODO: make it changeable
    int numBlocksNormalise = ((numMagnitudes/numMagnitudesPerThreadNormalise) + numThreadsNormalise - 1)/ numThreadsNormalise;
    //printf("numBlocksNormalise: %d\n", numBlocksNormalise);
    //printf("numMagnitudes: %d\n", numMagnitudes);
    
    if (debug == 1) {
        printf("Calling medianOfMediansNormalisation with %d blocks and %d threads per block\n", numBlocksNormalise, numThreadsNormalise);
    }

    // Take the magnitude of the complex numbers
    float* magnitudeSquaredArray;
    cudaMalloc((void**)&magnitudeSquaredArray, sizeof(float)*numMagnitudes);

    int numThreadsMagnitude = 1024;
    int numBlocksMagnitude = (numMagnitudes + numThreadsMagnitude - 1)/ numThreadsMagnitude;
    
    if (debug == 1) {
        printf("Calling magnitudeSquared with %d blocks and %d threads per block\n", numBlocksMagnitude, numThreadsMagnitude);
    }

    float* decimatedArrayBy2;
    float* decimatedArrayBy3;
    float* decimatedArrayBy4;
    cudaMalloc((void**)&decimatedArrayBy2, sizeof(float)*numMagnitudes/2);
    cudaMalloc((void**)&decimatedArrayBy3, sizeof(float)*numMagnitudes/3);
    cudaMalloc((void**)&decimatedArrayBy4, sizeof(float)*numMagnitudes/4);

    int numThreadsDecimate = 256;
    int numBlocksDecimate = (numMagnitudes/2 + numThreadsDecimate - 1)/ numThreadsDecimate;


    if (debug == 1) {
        printf("Calling decimateHarmonics with %d blocks and %d threads per block\n", numBlocksDecimate, numThreadsDecimate);
    }


    cudaStream_t stream1, stream2, stream3, stream4;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    int numThreadsBoxcar = 256;
    int numBlocksBoxcar1 = (numMagnitudes + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar2 = (numMagnitudes/2 + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar3 = (numMagnitudes/3 + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar4 = (numMagnitudes/4 + numThreadsBoxcar - 1)/ numThreadsBoxcar;

    candidate* globalCandidateArray1;
    candidate* globalCandidateArray2;
    candidate* globalCandidateArray3;
    candidate* globalCandidateArray4;

    int zmax = 256;
    int numCandidatesPerBlock = 1;

    int i = 1;
    while (i <= zmax){
        i *= 2;
        numCandidatesPerBlock += 1;
    }

    cudaMalloc((void**)&globalCandidateArray1, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1);
    cudaMalloc((void**)&globalCandidateArray2, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2);
    cudaMalloc((void**)&globalCandidateArray3, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3);
    cudaMalloc((void**)&globalCandidateArray4, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4);

    cudaMemset(globalCandidateArray1, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1);
    cudaMemset(globalCandidateArray2, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2);
    cudaMemset(globalCandidateArray3, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3);
    cudaMemset(globalCandidateArray4, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4);
    
    if (debug == 1) {
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar1, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar2, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar3, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar4, numThreadsBoxcar);
    }

    int numThreadsLogp = 256;
    int numBlocksLogp1 = (numBlocksBoxcar1*numCandidatesPerBlock + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp2 = (numBlocksBoxcar2*numCandidatesPerBlock + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp3 = (numBlocksBoxcar3*numCandidatesPerBlock + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp4 = (numBlocksBoxcar4*numCandidatesPerBlock + numThreadsLogp - 1)/ numThreadsLogp;

    if (debug == 1) {
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp1, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp2, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp3, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp4, numThreadsLogp);
    }


    for(int repeat = 0; repeat < 100000; repeat++){

        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaEventCreate(&overallGPUStart);
        cudaEventCreate(&overallGPUStop);
        cudaEventRecord(overallGPUStart);

        separateRealAndImaginaryComponents<<<numBlocksSeparate, numThreadsSeparate>>>((float2*)rawDataDevice, realDataDevice, imaginaryDataDevice, numMagnitudes);

        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Separating complex components took:     %f ms\n", milliseconds);


        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(realDataDevice);
        medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(imaginaryDataDevice);
        cudaDeviceSynchronize();

        //copyDeviceArrayToHostAndPrint(realDataDevice, numMagnitudes);
        
        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Normalisation took:                     %f ms\n", milliseconds);

        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        magnitudeSquared<<<numBlocksMagnitude, numThreadsMagnitude>>>(realDataDevice, imaginaryDataDevice, magnitudeSquaredArray, numMagnitudes);
        cudaDeviceSynchronize();
        
        //copyDeviceArrayToHostAndPrint(magnitudeSquaredArray, numMagnitudes);
        //copyDeviceArrayToHostAndSaveToFile(magnitudeSquaredArray, numMagnitudes, "magnitudeSquaredArray.csv");

        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Magnitude took:                         %f ms\n", milliseconds);

        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        decimateHarmonics<<<numBlocksDecimate, numThreadsDecimate>>>(magnitudeSquaredArray, decimatedArrayBy2, decimatedArrayBy3, decimatedArrayBy4, numMagnitudes);
        cudaDeviceSynchronize();

        //copyDeviceArrayToHostAndSaveToFile(decimatedArrayBy4, numMagnitudes/4, "decimatedArrayBy4.csv");
        
        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Decimation took:                        %f ms\n", milliseconds);

        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        boxcarFilterArray<<<numBlocksBoxcar1, numThreadsBoxcar, 0, stream1>>>(magnitudeSquaredArray, globalCandidateArray1, 1, numMagnitudes, numCandidatesPerBlock);
        boxcarFilterArray<<<numBlocksBoxcar2, numThreadsBoxcar, 0, stream2>>>(decimatedArrayBy2, globalCandidateArray2, 2, numMagnitudes/2, numCandidatesPerBlock);
        boxcarFilterArray<<<numBlocksBoxcar3, numThreadsBoxcar, 0, stream3>>>(decimatedArrayBy3, globalCandidateArray3, 3, numMagnitudes/3, numCandidatesPerBlock);
        boxcarFilterArray<<<numBlocksBoxcar4, numThreadsBoxcar, 0, stream4>>>(decimatedArrayBy4, globalCandidateArray4, 4, numMagnitudes/4, numCandidatesPerBlock);
        cudaDeviceSynchronize();

        //copyDeviceCandidateArrayToHostAndPrint(globalCandidateArray4,numCandidatesPerBlock*numBlocksBoxcar4);

        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Boxcar filtering took:                  %f ms\n", milliseconds);

        // start timing
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        calculateLogp<<<numBlocksLogp1, numThreadsLogp, 0, stream1>>>(globalCandidateArray1, numBlocksBoxcar1*numCandidatesPerBlock, 1);
        calculateLogp<<<numBlocksLogp2, numThreadsLogp, 0, stream2>>>(globalCandidateArray2, numBlocksBoxcar2*numCandidatesPerBlock, 3);
        calculateLogp<<<numBlocksLogp3, numThreadsLogp, 0, stream3>>>(globalCandidateArray3, numBlocksBoxcar3*numCandidatesPerBlock, 6);
        calculateLogp<<<numBlocksLogp4, numThreadsLogp, 0, stream4>>>(globalCandidateArray4, numBlocksBoxcar4*numCandidatesPerBlock, 10);
        cudaDeviceSynchronize();

        //copyDeviceCandidateArrayToHostAndPrint(globalCandidateArray1, numCandidatesPerBlock*numBlocksBoxcar1);
        
        // stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Logp time taken:                        %f ms\n", milliseconds);

        // stop overall GPU timing
        cudaEventRecord(overallGPUStop);
        cudaEventSynchronize(overallGPUStop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, overallGPUStart, overallGPUStop);
        printf("Overall GPU time taken:                 %f ms\n", milliseconds);

    }

    free(rawData);
    cudaFree(rawDataDevice);


    // check last cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;
}