#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

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
  

    for (int upperThreadIndex = 1024; upperThreadIndex > 0; upperThreadIndex >>=2){
        if(localThreadIndex < upperThreadIndex){
            a = medianArray[localThreadIndex];
            b = medianArray[localThreadIndex+upperThreadIndex];
            c = medianArray[localThreadIndex+upperThreadIndex*2];
            d = medianArray[localThreadIndex+upperThreadIndex*3];
            min = fminf(fminf(fminf(a,b),c),d);
            max = fmaxf(fmaxf(fmaxf(a,b),c),d);
            medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        }
        __syncthreads();
    }


    median = medianArray[0];
    __syncthreads();

    madArray[localThreadIndex] = fabsf(madArray[localThreadIndex] - median);
    madArray[localThreadIndex + 1024] = fabsf(madArray[localThreadIndex + 1024] - median);
    madArray[localThreadIndex + 2048] = fabsf(madArray[localThreadIndex + 2048] - median);
    madArray[localThreadIndex + 3072] = fabsf(madArray[localThreadIndex + 3072] - median);

    __syncthreads();
    
    for (int upperThreadIndex = 1024; upperThreadIndex > 0; upperThreadIndex >>=2){
        if(localThreadIndex < upperThreadIndex){
            a = madArray[localThreadIndex];
            b = madArray[localThreadIndex+upperThreadIndex];
            c = madArray[localThreadIndex+upperThreadIndex*2];
            d = madArray[localThreadIndex+upperThreadIndex*3];
            min = fminf(fminf(fminf(a,b),c),d);
            max = fmaxf(fmaxf(fmaxf(a,b),c),d);
            madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        }
        __syncthreads();
    }
    
    mad =  madArray[0]*1.4826;
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

// I WANT TO __FORCEINLINE__ THIS FUNCTION BUT APPARENTLY YOU CAN'T INLINE FUNCTIONS WITH A __SYNCTHREADS() IN
__device__ void searchAndUpdate(float* sumArray, power_index_struct* searchArray, candidate* localCandidateArray, int z, int outputCounter, int localThreadIndex, int globalThreadIndex, int numharm){
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
        localCandidateArray[outputCounter].r = searchArray[0].index;
        localCandidateArray[outputCounter].z = z;
        localCandidateArray[outputCounter].logp = 0.0f;
        localCandidateArray[outputCounter].numharm = numharm;
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
            searchAndUpdate(sumArray, searchArray, localCandidateArray, z, outputCounter, localThreadIndex, globalThreadIndex, numharm);
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
    size_t itemsRead = fread(rawData, sizeof(float), numFloats, f);
    if (itemsRead != numFloats) {
        // Handle error: not all items were read
        fprintf(stderr, "Error reading file: only %zu out of %zu items read\n", itemsRead, numFloats);
        // You might want to take additional action here, like exiting the function or the program
    }
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

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaEventCreate(&overallGPUStart);
    cudaEventCreate(&overallGPUStop);
    cudaEventRecord(overallGPUStart);

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

    // Normalise the real and imaginary parts

    int numMagnitudesPerThreadNormalise = 4;
    int numThreadsNormalise = 1024; // DO NOT CHANGE THIS, TODO: make it changeable
    int numBlocksNormalise = ((numMagnitudes/numMagnitudesPerThreadNormalise) + numThreadsNormalise - 1)/ numThreadsNormalise;
    //printf("numBlocksNormalise: %d\n", numBlocksNormalise);
    //printf("numMagnitudes: %d\n", numMagnitudes);
    
    if (debug == 1) {
        printf("Calling medianOfMediansNormalisation with %d blocks and %d threads per block\n", numBlocksNormalise, numThreadsNormalise);
    }
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

    // Take the magnitude of the complex numbers
    float* magnitudeSquaredArray;
    cudaMalloc((void**)&magnitudeSquaredArray, sizeof(float)*numMagnitudes);

    int numThreadsMagnitude = 1024;
    int numBlocksMagnitude = (numMagnitudes + numThreadsMagnitude - 1)/ numThreadsMagnitude;
    
    if (debug == 1) {
        printf("Calling magnitudeSquared with %d blocks and %d threads per block\n", numBlocksMagnitude, numThreadsMagnitude);
    }
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

    // start chrono timer for writing output file
    start_chrono = std::chrono::high_resolution_clock::now();

    candidate* hostCandidateArray1;
    candidate* hostCandidateArray2;
    candidate* hostCandidateArray3;
    candidate* hostCandidateArray4;

    hostCandidateArray1 = (candidate*)malloc(sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1);
    hostCandidateArray2 = (candidate*)malloc(sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2);
    hostCandidateArray3 = (candidate*)malloc(sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3);
    hostCandidateArray4 = (candidate*)malloc(sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4);

    cudaMemcpy(hostCandidateArray1, globalCandidateArray1, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray2, globalCandidateArray2, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray3, globalCandidateArray3, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray4, globalCandidateArray4, sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar4, cudaMemcpyDeviceToHost);

    char outputFilename[256];
    size_t filepathLen = strlen(filepath);
    
    if (filepathLen > 4 && strcmp(filepath + filepathLen - 4, ".fft") == 0) {
        // Copy the filepath without the last 4 characters
        snprintf(outputFilename, sizeof(outputFilename), "%.*s", (int)(filepathLen - 4), filepath);
    } else {
        // If the file doesn't end with .fft, just copy the whole filepath
        snprintf(outputFilename, sizeof(outputFilename), "%s", filepath);
    }
    
    // Append the new extension
    strncat(outputFilename, ".gpucand", sizeof(outputFilename) - strlen(outputFilename) - 1);

    // write the candidates to a csv file with a header line
    //FILE *csvFile = fopen("gpucandidates.csv", "w");
    FILE *csvFile = fopen(outputFilename, "w");
    fprintf(csvFile, "sigma,logp,r,z,power,numharm\n");

    float logpThreshold = -10;



    candidate* finalCandidateArray = (candidate*)malloc(sizeof(candidate)*numCandidatesPerBlock*numBlocksBoxcar1+numCandidatesPerBlock*numBlocksBoxcar2+numCandidatesPerBlock*numBlocksBoxcar3+numCandidatesPerBlock*numBlocksBoxcar4);
    int candidateCounter = 0;

    for (int i = 0; i < numBlocksBoxcar1*numCandidatesPerBlock; i++){
        if (hostCandidateArray1[i].logp < logpThreshold){
            if (hostCandidateArray1[i].r != 0){
                if (hostCandidateArray1[i].z != 0){
                    finalCandidateArray[candidateCounter] = hostCandidateArray1[i];
                    candidateCounter+=1;
                }
            }
        }
    }
    
    for (int i = 0; i < numBlocksBoxcar2*numCandidatesPerBlock; i++){
        if (hostCandidateArray2[i].logp < logpThreshold){
            if (hostCandidateArray2[i].r != 0){
                if (hostCandidateArray2[i].z != 0){
                    finalCandidateArray[candidateCounter] = hostCandidateArray2[i];
                    candidateCounter+=1;
                }
            }
        }
    }

    for (int i = 0; i < numBlocksBoxcar3*numCandidatesPerBlock; i++){
        if (hostCandidateArray3[i].logp < logpThreshold){
            if (hostCandidateArray3[i].r != 0){
                if (hostCandidateArray3[i].z != 0){
                    finalCandidateArray[candidateCounter] = hostCandidateArray3[i];
                    candidateCounter+=1;
                }
            }
        }
    }

    
    for (int i = 0; i < numBlocksBoxcar4*numCandidatesPerBlock; i++){
        if (hostCandidateArray4[i].logp < logpThreshold){
            if (hostCandidateArray4[i].r != 0){
                if (hostCandidateArray4[i].z != 0){
                    finalCandidateArray[candidateCounter] = hostCandidateArray4[i];
                    candidateCounter+=1;
                }
            }
        }
    }
    

    // use quicksort to sort the final candidate array by logp (ascending order)
    qsort(finalCandidateArray, candidateCounter, sizeof(candidate), compareCandidatesByLogp);

    for (int i = 0; i < candidateCounter; i++){
        fprintf(csvFile, "%lf, %f,%d,%d,%f,%d\n", equivalent_gaussian_sigma((double) finalCandidateArray[i].logp), finalCandidateArray[i].logp, finalCandidateArray[i].r, finalCandidateArray[i].z, finalCandidateArray[i].power, finalCandidateArray[i].numharm);
    }


    fclose(csvFile);

    // stop chrono timer for writing output file
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Writing output file took:               %f ms\n", (float)duration.count());


    free(rawData);
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

    // check last cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;
}