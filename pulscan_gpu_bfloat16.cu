#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cufft.h>
#include <cuda_bf16.h>

extern "C" {
#include "localcdflib.h"
}

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

double extended_equiv_gaussian_sigma(double logp)
/*
  extended_equiv_gaussian_sigma(double logp):
      Return the equivalent gaussian sigma corresponding to the 
          natural log of the cumulative gaussian probability logp.
          In other words, return x, such that Q(x) = p, where Q(x)
          is the cumulative normal distribution.  This version uses
          the rational approximation from Abramowitz and Stegun,
          eqn 26.2.23.  Using the log(P) as input gives a much
          extended range.
*/
{
    double t, num, denom;

    t = sqrt(-2.0 * logp);
    num = 2.515517 + t * (0.802853 + t * 0.010328);
    denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    return t - num / denom;
}

double equivalent_gaussian_sigma(double logp)
/* Return the approximate significance in Gaussian sigmas */
/* corresponding to a natural log probability logp        */
{
    double x;

    if (logp < -600.0) {
        x = extended_equiv_gaussian_sigma(logp);
    } else {
        int which, status;
        double p, q, bound, mean = 0.0, sd = 1.0;
        q = exp(logp);
        p = 1.0 - q;
        which = 2;
        status = 0;
        /* Convert to a sigma */
        cdfnor(&which, &p, &q, &x, &mean, &sd, &status, &bound);
        if (status) {
            if (status == -2) {
                x = 0.0;
            } else if (status == -3) {
                x = 38.5;
            } else {
                printf("\nError in cdfnor() (candidate_sigma()):\n");
                printf("   status = %d, bound = %g\n", status, bound);
                printf("   p = %g, q = %g, x = %g, mean = %g, sd = %g\n\n",
                       p, q, x, mean, sd);
                exit(1);
            }
        }
    }
    if (x < 0.0)
        return 0.0;
    else
        return x;
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
}

__global__ void separateRealAndImaginaryComponents(__nv_bfloat162* rawDataDevice, __nv_bfloat16* realData, __nv_bfloat16* imaginaryData, long numComplexFloats){
    long globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numComplexFloats){
        __nv_bfloat162 currentValue = rawDataDevice[globalThreadIndex];
        realData[globalThreadIndex] = currentValue.x;
        imaginaryData[globalThreadIndex] = currentValue.y;
    }
}



__global__ void medianOfMediansNormalisation(__nv_bfloat16* globalArray) {
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
    __shared__ __nv_bfloat16 medianArray[4096];
    __shared__ __nv_bfloat16 madArray[4096];
    __shared__ __nv_bfloat16 normalisedArray[4096];

    //int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;
    int globalArrayIndex = blockDim.x*blockIdx.x*4+threadIdx.x;

    __nv_bfloat16 median;
    __nv_bfloat16 mad;

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

    __nv_bfloat16 a,b,c,d,min,max;
  

    for (int upperThreadIndex = 1024; upperThreadIndex > 0; upperThreadIndex >>=2){
        if(localThreadIndex < upperThreadIndex){
            a = medianArray[localThreadIndex];
            b = medianArray[localThreadIndex+upperThreadIndex];
            c = medianArray[localThreadIndex+upperThreadIndex*2];
            d = medianArray[localThreadIndex+upperThreadIndex*3];
            min = __hmin(__hmin(__hmin(a,b),c),d);
            max = __hmax(__hmax(__hmax(a,b),c),d);
            medianArray[localThreadIndex] = (a+b+c+d-min-max)*__float2bfloat16(0.5);
        }
        __syncthreads();
    }


    median = medianArray[0];
    __syncthreads();

    madArray[localThreadIndex] = __habs(madArray[localThreadIndex] - median);
    madArray[localThreadIndex + 1024] = __habs(madArray[localThreadIndex + 1024] - median);
    madArray[localThreadIndex + 2048] = __habs(madArray[localThreadIndex + 2048] - median);
    madArray[localThreadIndex + 3072] = __habs(madArray[localThreadIndex + 3072] - median);

    __syncthreads();
    
    for (int upperThreadIndex = 1024; upperThreadIndex > 0; upperThreadIndex >>=2){
        if(localThreadIndex < upperThreadIndex){
            a = madArray[localThreadIndex];
            b = madArray[localThreadIndex+upperThreadIndex];
            c = madArray[localThreadIndex+upperThreadIndex*2];
            d = madArray[localThreadIndex+upperThreadIndex*3];
            min = __hmin(__hmin(__hmin(a,b),c),d);
            max = __hmax(__hmax(__hmax(a,b),c),d);
            madArray[localThreadIndex] = (a+b+c+d-min-max)*__float2bfloat16(0.5);
        }
        __syncthreads();
    }
    
    mad =  madArray[0]*__float2bfloat16(1.4826);
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

}


__global__ void magnitudeSquared(__nv_bfloat16* realData, __nv_bfloat16* imaginaryData, __nv_bfloat16* magnitudeSquaredArray, long numFloats){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numFloats){
        __nv_bfloat16 real = realData[globalThreadIndex];
        __nv_bfloat16 imaginary = imaginaryData[globalThreadIndex];
        magnitudeSquaredArray[globalThreadIndex] = real*real + imaginary*imaginary;

        // check for inf or nan
        //if (isinf(magnitudeSquaredArray[globalThreadIndex]) || isnan(magnitudeSquaredArray[globalThreadIndex])){
        //        printf("magnitudeSquaredArray[%ld] = %f\n", globalThreadIndex, magnitudeSquaredArray[globalThreadIndex]);
        //        printf("realData[%ld] = %f\n", globalThreadIndex, realData[globalThreadIndex]);
        //        printf("imaginaryData[%ld] = %f\n", globalThreadIndex, imaginaryData[globalThreadIndex]);
        //}
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

__global__ void decimateHarmonics(__nv_bfloat16* magnitudeSquaredArray, __nv_bfloat16* decimatedArray2, __nv_bfloat16* decimatedArray3, __nv_bfloat16* decimatedArray4, long numMagnitudes){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;

    __nv_bfloat16 fundamental;
    __nv_bfloat16 harmonic1a, harmonic1b;
    __nv_bfloat16 harmonic2a, harmonic2b, harmonic2c;
    __nv_bfloat16 harmonic3a, harmonic3b, harmonic3c, harmonic3d;

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

// I WANT TO __FORCEINLINE__ THIS FUNCTION BUT APPARENTLY YOU CAN'T INLINE FUNCTIONS WITH A __SYNCTHREADS() IN
__device__ void searchAndUpdate(float* sumArray, power_index_struct* searchArray, candidate* localCandidateArray, int z, int outputCounter, int localThreadIndex, int globalThreadIndex, int numharm){
    searchArray[localThreadIndex].power = sumArray[localThreadIndex];
    searchArray[localThreadIndex].index = localThreadIndex;
    for (int stride = blockDim.x / 2; stride>0; stride /= 2){
        if (localThreadIndex < stride){
            if (searchArray[localThreadIndex].power < searchArray[localThreadIndex + stride].power){
                searchArray[localThreadIndex] = searchArray[localThreadIndex + stride];
            }
        }
        __syncthreads();
    }
    if (localThreadIndex == 0){
        localCandidateArray[outputCounter].power = __bfloat162float(searchArray[0].power);
        localCandidateArray[outputCounter].r = blockIdx.x * blockDim.x + (int) searchArray[0].index;
        localCandidateArray[outputCounter].z = z;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
    }
}


__global__ void boxcarFilterArray(__nv_bfloat16* magnitudeSquaredArray, candidate* globalCandidateArray, int numharm, long numFloats, int numCandidatesPerBlock){
    __shared__ float lookupArray[512];
    __shared__ float sumArray[256];
    __shared__ power_index_struct searchArray[256];
    __shared__ candidate localCandidateArray[16]; //oversized, has to be greater than numCandidatesPerBlock

    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;

    lookupArray[localThreadIndex] = __bfloat162float(magnitudeSquaredArray[globalThreadIndex]);
    lookupArray[localThreadIndex + 256] = __bfloat162float(magnitudeSquaredArray[globalThreadIndex + 256]);
    //printf("lookupArray[%d] = %f\n", localThreadIndex, lookupArray[localThreadIndex]);
    __syncthreads();

    // initialise the sum array
    sumArray[localThreadIndex] = 0.0f;
    __syncthreads();

    // begin boxcar filtering
    // search at z = 0
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 0];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 0, 0, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 1
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 1];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 1, 1, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 2
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 2];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 2, 2, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 4
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 3];
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 4];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 4, 3, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 8
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 5];
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 6];
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 7];
    sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + 8];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 8, 4, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 16
    #pragma unroll
    for (int z = 9; z < 17; z++){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 16, 5, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 32
    #pragma unroll
    for (int z = 17; z < 33; z++){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 32, 6, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 64
    #pragma unroll
    for (int z = 33; z < 65; z++){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 64, 7, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 128
    #pragma unroll
    for (int z = 65; z < 129; z++){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 128, 8, localThreadIndex, globalThreadIndex, numharm);

    // search at z = 256
    #pragma unroll
    for (int z = 129; z < 257; z++){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 256, 9, localThreadIndex, globalThreadIndex, numharm);

    __syncthreads();

    if (localThreadIndex < numCandidatesPerBlock){
        globalCandidateArray[blockIdx.x*numCandidatesPerBlock+localThreadIndex] = localCandidateArray[localThreadIndex];
    }
}

/*__global__ void boxcarFilterArrayvec2(
    const __nv_bfloat16* __restrict__ magnitudeSquaredArray,
    candidate*           __restrict__ globalCandidateArray,
    int numharm,
    long numFloats,
    int numCandidatesPerBlock)
{
    // We will read 512 BF16 elements into shared memory:
    //   lookupArray[0..511]
    // blockDim.x = 256, so we want half the threads (128) to do the loads.
    //
    // Each loader thread fetches a __nv_bfloat162 from global memory (2 BF16s),
    // and places them into consecutive slots in lookupArray[].

    __shared__ __nv_bfloat16 lookupArray[512];       // For 512 BF16
    __shared__ __nv_bfloat16 sumArray[256];          // One sum per thread
    __shared__ power_index_struct searchArray[256];  // For maxima
    __shared__ candidate localCandidateArray[16];    // Must be ≥ numCandidatesPerBlock

    int tid = threadIdx.x;
    int blockBase = blockIdx.x * blockDim.x; // e.g. 256 * blockIdx.x

    // Only half the threads (128 if blockDim.x=256) will load from global memory
    int halfThreads = blockDim.x / 2;  // 128

    if (tid < halfThreads)
    {
        // Two 16-bit BF16 elements => one 32-bit read
        // 1) Load the first pair
        int loadIndex = blockBase + 2 * tid; // offset for pairs
        if (loadIndex + 1 < numFloats) // boundary check
        {
            // Reinterpret pointer to the BF16 array as a pointer to __nv_bfloat162
            const __nv_bfloat162* vec2ptr =
                reinterpret_cast<const __nv_bfloat162*>(magnitudeSquaredArray + loadIndex);
            __nv_bfloat162 val2 = vec2ptr[0]; // One 32-bit load => 2 BF16

            // Decompose into two BF16 elements and store in shared memory
            lookupArray[2 * tid + 0] = val2.x;
            lookupArray[2 * tid + 1] = val2.y;
        }
        else {
            // If out of range, write zeros
            lookupArray[2 * tid + 0] = __float2bfloat16(0.0f);
            lookupArray[2 * tid + 1] = __float2bfloat16(0.0f);
        }

        // 2) Load the second pair for the top half of the block, offset by +256 in shared
        int loadIndex2 = blockBase + 256 + 2 * tid;
        if (loadIndex2 + 1 < numFloats)
        {
            const __nv_bfloat162* vec2ptr2 =
                reinterpret_cast<const __nv_bfloat162*>(magnitudeSquaredArray + loadIndex2);
            __nv_bfloat162 val2b = vec2ptr2[0];

            lookupArray[256 + 2 * tid + 0] = val2b.x;
            lookupArray[256 + 2 * tid + 1] = val2b.y;
        }
        else {
            lookupArray[256 + 2 * tid + 0] = __float2bfloat16(0.0f);
            lookupArray[256 + 2 * tid + 1] = __float2bfloat16(0.0f);
        }
    }

    __syncthreads();

    // Now all 256 threads can safely read from lookupArray[].
    // Initialize sumArray for each thread
    sumArray[tid] = __float2bfloat16(0.0f);
    __syncthreads();

    // The rest is your original “boxcar” searching logic:
    //   sum at z=0, then z=1, etc.
    //   The only difference is that we do not individually load from
    //   magnitudeSquaredArray[] in each thread. Instead we read from
    //   lookupArray[] that was already populated.

    // -- search at z = 0
    sumArray[tid] += lookupArray[tid + 0];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 0, 0, tid, blockBase + tid, numharm);

    // -- search at z = 1
    sumArray[tid] += lookupArray[tid + 1];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 1, 1, tid, blockBase + tid, numharm);

    // -- search at z = 2
    sumArray[tid] += lookupArray[tid + 2];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 2, 2, tid, blockBase + tid, numharm);

    // -- search at z = 4
    sumArray[tid] += lookupArray[tid + 3];
    sumArray[tid] += lookupArray[tid + 4];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 4, 3, tid, blockBase + tid, numharm);

    // -- search at z = 8
    sumArray[tid] += lookupArray[tid + 5];
    sumArray[tid] += lookupArray[tid + 6];
    sumArray[tid] += lookupArray[tid + 7];
    sumArray[tid] += lookupArray[tid + 8];
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 8, 4, tid, blockBase + tid, numharm);

    // -- search at z = 16
    #pragma unroll
    for (int z = 9; z < 17; z++){
        sumArray[tid] += lookupArray[tid + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 16, 5, tid, blockBase + tid, numharm);

    // -- search at z = 32
    #pragma unroll
    for (int z = 17; z < 33; z++){
        sumArray[tid] += lookupArray[tid + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 32, 6, tid, blockBase + tid, numharm);

    // -- search at z = 64
    #pragma unroll
    for (int z = 33; z < 65; z++){
        sumArray[tid] += lookupArray[tid + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 64, 7, tid, blockBase + tid, numharm);

    // -- search at z = 128
    #pragma unroll
    for (int z = 65; z < 129; z++){
        sumArray[tid] += lookupArray[tid + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 128, 8, tid, blockBase + tid, numharm);

    // -- search at z = 256
    #pragma unroll
    for (int z = 129; z < 257; z++){
        sumArray[tid] += lookupArray[tid + z];
    }
    __syncthreads();
    searchAndUpdate(sumArray, searchArray, localCandidateArray, 256, 9, tid, blockBase + tid, numharm);

    __syncthreads();

    // Finally, store the block’s local candidates back to global
    if (tid < numCandidatesPerBlock) {
        globalCandidateArray[blockIdx.x * numCandidatesPerBlock + tid] =
            localCandidateArray[tid];
    }
}*/


__global__ void calculateLogp(candidate* globalCandidateArray, long numCandidates, int numSum){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numCandidates){
        double logp = power_to_logp(globalCandidateArray[globalThreadIndex].power,globalCandidateArray[globalThreadIndex].z*numSum*2);
        globalCandidateArray[globalThreadIndex].logp = (float) logp;
    }
}

__global__ void convertFP32ArrayToBF16Array(float* inputArray, __nv_bfloat16* outputArray, long numFloats){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numFloats){
        outputArray[globalThreadIndex] = __float2bfloat16(inputArray[globalThreadIndex]);
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

// (Discard DC kernel definition)
__global__ void discardDC_kernel(const float* __restrict__ d_in,
                                 float* __restrict__ d_out,
                                 int outCount)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < outCount-2) {
        d_out[idx] = d_in[idx + 2]; 
    }
}


long readDatAndDiscardDC(const char* filepath, float** d_out)
{
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open .dat file: %s\n", filepath);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fileSizeBytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    long N = fileSizeBytes / sizeof(float);
    if (N < 2) {
        fprintf(stderr, "Not enough data in .dat file.\n");
        fclose(f);
        return -1;
    }

    // Optionally align down to multiple of 8192 if desired:
    // N -= (N % 8192);

    float* h_timeData = (float*)malloc(N*sizeof(float));
    size_t itemsRead = fread(h_timeData, sizeof(float), N, f);
    fclose(f);
    if (itemsRead != (size_t)N) {
        fprintf(stderr, "Error reading .dat: only %zu of %ld floats read\n",
                itemsRead, N);
        free(h_timeData);
        return -1;
    }

    printf("N+2 = %ld\n", N+2);

    float* d_timeData = NULL;
    cudaMalloc((void**)&d_timeData, N*sizeof(float));
    cudaMemcpy(d_timeData, h_timeData, N*sizeof(float), cudaMemcpyHostToDevice);
    free(h_timeData);

    float* d_freqTemp = NULL; 
    cudaMalloc((void**)&d_freqTemp, (N+2)*sizeof(float)); // R2C => N+2 floats

    cufftHandle plan;
    cufftPlan1d(&plan, (int)N, CUFFT_R2C, 1);
    cufftResult res = cufftExecR2C(plan, (cufftReal*)d_timeData, (cufftComplex*)d_freqTemp);
    cufftDestroy(plan);
    cudaFree(d_timeData);
    if (res != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecR2C failed.\n");
        cudaFree(d_freqTemp);
        return -1;
    }


    // Allocate final array for "no-DC" => length N floats
    float* d_freqNoDC = NULL;
    cudaMalloc((void**)&d_freqNoDC, N*sizeof(float));

    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1)/blockSize);
    discardDC_kernel<<<gridSize, blockSize>>>(d_freqTemp, d_freqNoDC, (int)N);

    cudaFree(d_freqTemp);

    *d_out = d_freqNoDC;



    return N; // the final array has N floats (i.e. (N/2) complex bins)
}


// Pulscan ASCII
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
    
    // Detect file extension
    size_t filepathLen = strlen(filepath);
    bool isFFT = false, isDAT = false;
    if (filepathLen > 4) {
        const char* ext = filepath + (filepathLen - 4);
        if (strcmp(ext, ".fft") == 0) {
            isFFT = true;
        } else if (strcmp(ext, ".dat") == 0) {
            isDAT = true;
        }
    }
    if (!isFFT && !isDAT) {
        printf("Error: input file must end with .fft or .dat\n");
        return 1;
    }

    float* rawDataDevice = NULL;
    size_t numFloats = 0; // total floats in rawDataDevice

    if (isFFT) {
        FILE *f = fopen(filepath, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open .fft file: %s\n", filepath);
            return 1;
        }

        fseek(f, 0, SEEK_END);
        size_t filesize = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        numFloats = filesize / sizeof(float);

        // Cap the filesize at the nearest lower factor of 8192
        numFloats = numFloats - (numFloats % 8192);

        float* rawDataHost = (float*) malloc(sizeof(float)*numFloats);
        size_t itemsRead = fread(rawDataHost, sizeof(float), numFloats, f);
        fclose(f);
        if (itemsRead != numFloats) {
            fprintf(stderr, "Error reading .fft file: only %zu of %zu items read\n",
                    itemsRead, numFloats);
        }
        end_chrono = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
        printf("Reading file took:                      %f ms\n", (float)duration.count());

        // Copy raw .fft data to GPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMalloc((void**)&rawDataDevice, sizeof(float)*numFloats);
        cudaMemcpy(rawDataDevice, rawDataHost, sizeof(float)*numFloats, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Copying data to GPU took:               %f ms\n", milliseconds);

        free(rawDataHost);

    } else {

        // read the .dat file, run R2C, discard DC => get an array of length N floats
        // This also includes the time to copy to GPU, so let's time it
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        auto start_chrono = std::chrono::high_resolution_clock::now();

        long N = readDatAndDiscardDC(filepath, &rawDataDevice);

        auto end_chrono = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
        printf("Reading file took:                      %f ms\n", (float)duration.count());

        if (N <= 0) {
            fprintf(stderr, "readDatAndDiscardDC failed.\n");
            return 1;
        }
        numFloats = (size_t)N;  // final # of floats in rawDataDevice

        // Cap the floats at the nearest lower factor of 8192
        numFloats = numFloats - (numFloats % 8192);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Performing R2C & discarding DC took:    %f ms\n", milliseconds);
    }

    __nv_bfloat16* rawDataDevice_bf16;
    cudaMalloc((void**)&rawDataDevice_bf16, sizeof(__nv_bfloat16)*numFloats);

    // Now we have:
    //   rawDataDevice: interleaved complex floats
    //   numFloats: total floats in that array
    //
    // The remainder of the pipeline is unchanged.




    int numMagnitudes = numFloats/2;  // each complex bin has 2 floats
    printf("Number of magnitude bins:               %d\n", numMagnitudes);

    // 1) Separate real & imaginary


    int numThreadsSeparate = 256;
    int numBlocksSeparate = (numMagnitudes + numThreadsSeparate - 1)/ numThreadsSeparate;

    int numThreadsConvert = 256;
    int numBlocksConvert = (numFloats + numThreadsConvert - 1)/ numThreadsConvert;

    long numFloats2 = numMagnitudes/2;
    numFloats2 = numFloats2 - (numFloats2 % 8192);

    long numFloats3 = numMagnitudes/3;
    numFloats3 = numFloats3 - (numFloats3 % 8192);

    long numFloats4 = numMagnitudes/4;
    numFloats4 = numFloats4 - (numFloats4 % 8192);


    __nv_bfloat16* realDataDevice;
    __nv_bfloat16* imaginaryDataDevice;
    cudaMalloc((void**)&realDataDevice, sizeof(__nv_bfloat16)*numMagnitudes);
    cudaMalloc((void**)&imaginaryDataDevice, sizeof(__nv_bfloat16)*numMagnitudes);

    __nv_bfloat16* magnitudeSquaredArray;
    cudaMalloc((void**)&magnitudeSquaredArray, sizeof(__nv_bfloat16)*numMagnitudes);

    __nv_bfloat16* decimatedArrayBy2;
    __nv_bfloat16* decimatedArrayBy3;
    __nv_bfloat16* decimatedArrayBy4;
    cudaMalloc((void**)&decimatedArrayBy2, sizeof(__nv_bfloat16)* (numMagnitudes/2));
    cudaMalloc((void**)&decimatedArrayBy3, sizeof(__nv_bfloat16)* (numMagnitudes/3));
    cudaMalloc((void**)&decimatedArrayBy4, sizeof(__nv_bfloat16)* (numMagnitudes/4));

    candidate* globalCandidateArray1;
    candidate* globalCandidateArray2;
    candidate* globalCandidateArray3;
    candidate* globalCandidateArray4;

    int zmax = 256;
    int numCandidatesPerBlock = 1;
    int i = 1;
    while (i <= zmax){
        i *= 2;
        numCandidatesPerBlock++;
    }



    int numThreadsNormalise = 1024; // must be 1024 for kernel
    int numMagnitudesPerThreadNormalise = 4;
    int numBlocksNormalise = ((numMagnitudes/numMagnitudesPerThreadNormalise)
                              + numThreadsNormalise - 1)/ numThreadsNormalise;

    int numThreadsDecimate = 256;
    int numBlocksDecimate = (numMagnitudes/2 + numThreadsDecimate - 1)/ numThreadsDecimate;

    int numThreadsBoxcar = 256;
    int numBlocksBoxcar1 = (numMagnitudes + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar2 = ((numFloats2) + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar3 = ((numFloats3) + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar4 = ((numFloats4) + numThreadsBoxcar - 1)/ numThreadsBoxcar;

    int numThreadsMagnitude = 1024;
    int numBlocksMagnitude = (numMagnitudes + numThreadsMagnitude - 1)/ numThreadsMagnitude;
    
    int numThreadsLogp = 256;
    int totalCands1 = numBlocksBoxcar1 * numCandidatesPerBlock;
    int totalCands2 = numBlocksBoxcar2 * numCandidatesPerBlock;
    int totalCands3 = numBlocksBoxcar3 * numCandidatesPerBlock;
    int totalCands4 = numBlocksBoxcar4 * numCandidatesPerBlock;

    int numBlocksLogp1 = (totalCands1 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp2 = (totalCands2 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp3 = (totalCands3 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp4 = (totalCands4 + numThreadsLogp - 1)/ numThreadsLogp;

    cudaMalloc((void**)&globalCandidateArray1, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1);
    cudaMalloc((void**)&globalCandidateArray2, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2);
    cudaMalloc((void**)&globalCandidateArray3, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3);
    cudaMalloc((void**)&globalCandidateArray4, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4);

    cudaMemset(globalCandidateArray1, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1);
    cudaMemset(globalCandidateArray2, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2);
    cudaMemset(globalCandidateArray3, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3);
    cudaMemset(globalCandidateArray4, 0, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4);
    
    // Start measuring GPU pipeline
    cudaEvent_t start, stop, overallGPUStart, overallGPUStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float millisecondsSeparate = 0;
    float millisecondsNormalise = 0;
    float millisecondsMagnitude = 0;
    float millisecondsDecimate = 0;
    float millisecondsBoxcar = 0;
    float millisecondsLogp = 0;
    float millisecondsOverall = 0;

    // convert rawDataDevice to __nv_bfloat16


    convertFP32ArrayToBF16Array<<<numBlocksConvert, numThreadsConvert>>>(rawDataDevice, rawDataDevice_bf16, numFloats);


    cudaEventCreate(&overallGPUStart);
    cudaEventCreate(&overallGPUStop);
    cudaEventRecord(overallGPUStart);
    cudaEventRecord(start);
    separateRealAndImaginaryComponents<<<numBlocksSeparate, numThreadsSeparate>>>(
        (__nv_bfloat162*)rawDataDevice_bf16, realDataDevice, imaginaryDataDevice, numMagnitudes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsSeparate, start, stop);
    

    // 2) Normalise real and imaginary parts
    

    cudaEventRecord(start);
    medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(realDataDevice);
    medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(imaginaryDataDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsNormalise, start, stop);
    

    // 3) Magnitude

    cudaEventRecord(start);
    magnitudeSquared<<<numBlocksMagnitude, numThreadsMagnitude>>>(
        realDataDevice, imaginaryDataDevice, magnitudeSquaredArray, numMagnitudes);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&millisecondsMagnitude, start, stop);
    

    // 4) Decimate for harmonic summation

    cudaEventRecord(start);
    decimateHarmonics<<<numBlocksDecimate, numThreadsDecimate>>>(
        magnitudeSquaredArray, decimatedArrayBy2, decimatedArrayBy3, decimatedArrayBy4, 
        numMagnitudes);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsDecimate, start, stop);


    // 5) Boxcar filtering on each harmonic array (1,2,3,4)

    cudaEventRecord(start);

    // Launch boxcar for each harmonic array
    boxcarFilterArray<<<numBlocksBoxcar1, numThreadsBoxcar, 0>>>(
        magnitudeSquaredArray, globalCandidateArray1, 1, numMagnitudes, numCandidatesPerBlock);
    boxcarFilterArray<<<numBlocksBoxcar2, numThreadsBoxcar, 0>>>(
        decimatedArrayBy2, globalCandidateArray2, 2, numMagnitudes/2, numCandidatesPerBlock);
    boxcarFilterArray<<<numBlocksBoxcar3, numThreadsBoxcar, 0>>>(
        decimatedArrayBy3, globalCandidateArray3, 3, numMagnitudes/3, numCandidatesPerBlock);
    boxcarFilterArray<<<numBlocksBoxcar4, numThreadsBoxcar, 0>>>(
        decimatedArrayBy4, globalCandidateArray4, 4, numMagnitudes/4, numCandidatesPerBlock);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsBoxcar, start, stop);

    // 6) Calculate logp for each candidate

    cudaEventRecord(start);
    calculateLogp<<<numBlocksLogp1, numThreadsLogp, 0>>>(
        globalCandidateArray1, totalCands1, 1);
    calculateLogp<<<numBlocksLogp2, numThreadsLogp, 0>>>(
        globalCandidateArray2, totalCands2, 3);
    calculateLogp<<<numBlocksLogp3, numThreadsLogp, 0>>>(
        globalCandidateArray3, totalCands3, 6);
    calculateLogp<<<numBlocksLogp4, numThreadsLogp, 0>>>(
        globalCandidateArray4, totalCands4, 10);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsLogp, start, stop);

    // stop overall GPU timing
    cudaEventRecord(overallGPUStop);
    cudaEventSynchronize(overallGPUStop);
    cudaEventElapsedTime(&millisecondsOverall, overallGPUStart, overallGPUStop);


    printf("Separating complex components took:     %f ms\n", millisecondsSeparate);
    printf("Normalisation took:                     %f ms\n", millisecondsNormalise);
    printf("Magnitude took:                         %f ms\n", millisecondsMagnitude);
    printf("Decimation took:                        %f ms\n", millisecondsDecimate);
    printf("Boxcar filtering took:                  %f ms\n", millisecondsBoxcar);
    printf("Logp time taken:                        %f ms\n", millisecondsLogp);
    printf("Overall GPU time taken:                 %f ms\n", millisecondsOverall);

    // ----------------------------------------------------
    // Write output file
    // ----------------------------------------------------
    auto write_start = std::chrono::high_resolution_clock::now();

    candidate* hostCandidateArray1 = (candidate*)malloc(sizeof(candidate)*totalCands1);
    candidate* hostCandidateArray2 = (candidate*)malloc(sizeof(candidate)*totalCands2);
    candidate* hostCandidateArray3 = (candidate*)malloc(sizeof(candidate)*totalCands3);
    candidate* hostCandidateArray4 = (candidate*)malloc(sizeof(candidate)*totalCands4);

    cudaMemcpy(hostCandidateArray1, globalCandidateArray1, sizeof(candidate)*totalCands1, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray2, globalCandidateArray2, sizeof(candidate)*totalCands2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray3, globalCandidateArray3, sizeof(candidate)*totalCands3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray4, globalCandidateArray4, sizeof(candidate)*totalCands4, cudaMemcpyDeviceToHost);

    // Build output filename
    char outputFilename[256];
    snprintf(outputFilename, sizeof(outputFilename), "%s", filepath);
    if (filepathLen > 4) {
        // remove the .fft or .dat suffix
        outputFilename[filepathLen - 4] = '\0';
    }
    strncat(outputFilename, ".gpucand", sizeof(outputFilename) - strlen(outputFilename) - 1);

    FILE *csvFile = fopen(outputFilename, "w");
    fprintf(csvFile, "sigma,logp,r,z,power,numharm\n");

    float logpThreshold = -10;

    // Gather final candidates
    int totalSize = totalCands1 + totalCands2 + totalCands3 + totalCands4;
    candidate* finalCandidateArray = (candidate*)malloc(sizeof(candidate)*totalSize);

    int candidateCounter = 0;
    #define ADD_CANDIDATES_FROM(Array, Count) \
        for (int i = 0; i < (Count); i++){ \
            if ((Array)[i].logp < logpThreshold && (Array)[i].r != 0 && (Array)[i].z != 0) { \
                finalCandidateArray[candidateCounter++] = (Array)[i]; \
            } \
        }

    ADD_CANDIDATES_FROM(hostCandidateArray1, totalCands1);
    ADD_CANDIDATES_FROM(hostCandidateArray2, totalCands2);
    ADD_CANDIDATES_FROM(hostCandidateArray3, totalCands3);
    ADD_CANDIDATES_FROM(hostCandidateArray4, totalCands4);

    // sort by logp ascending
    qsort(finalCandidateArray, candidateCounter, sizeof(candidate), compareCandidatesByLogp);

    // write out
    for (int i = 0; i < candidateCounter; i++){
        double sig = equivalent_gaussian_sigma((double) finalCandidateArray[i].logp);
        fprintf(csvFile, "%lf,%f,%d,%d,%f,%d\n",
                sig, finalCandidateArray[i].logp,
                finalCandidateArray[i].r, finalCandidateArray[i].z,
                finalCandidateArray[i].power, finalCandidateArray[i].numharm);
    }
    fclose(csvFile);

    auto write_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);
    printf("Writing output file took:               %f ms\n", (float)duration.count());

    // Cleanup
    free(hostCandidateArray1);
    free(hostCandidateArray2);
    free(hostCandidateArray3);
    free(hostCandidateArray4);
    free(finalCandidateArray);

    cudaFree(rawDataDevice);
    cudaFree(realDataDevice);
    cudaFree(imaginaryDataDevice);
    cudaFree(magnitudeSquaredArray);
    cudaFree(decimatedArrayBy2);
    cudaFree(decimatedArrayBy3);
    cudaFree(decimatedArrayBy4);
    cudaFree(globalCandidateArray1);
    cudaFree(globalCandidateArray2);
    cudaFree(globalCandidateArray3);
    cudaFree(globalCandidateArray4);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;
}
