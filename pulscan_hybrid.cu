// Jack White 2023, jack.white@eng.ox.ac.uk

// This program reads in a .fft file produced by PRESTO realfft
// and computes the boxcar filter candidates for a range of boxcar widths, 0 to zmax (default 200)


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

extern "C" {
#include "localcdflib.h"
}

#include <cuda_runtime.h>
#include <cuda.h>

#define SPEED_OF_LIGHT 299792458.0

// ANSI Color Codes
#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"
#define ITALIC   "\033[3m"

typedef struct {
    double sigma;
    float power;
    long index;
    int z;
    int harmonic; 
} candidate_struct;

typedef struct {
    float power;
    int index;
    int z;
    int harmonic;
} candidate_struct_short;

typedef struct {
    float power;
    int index;
} power_index_struct;

int compare_candidate_structs_sigma(const void *a, const void *b) {
    candidate_struct *candidateA = (candidate_struct *)a;
    candidate_struct *candidateB = (candidate_struct *)b;
    if(candidateA->sigma > candidateB->sigma) return -1; // for descending order
    if(candidateA->sigma < candidateB->sigma) return 1;
    return 0;
}

float fdot_from_boxcar_width(int boxcar_width, float observation_time_seconds){
    return boxcar_width / (observation_time_seconds*observation_time_seconds);
}

float acceleration_from_fdot(float fdot, float frequency){
    return fdot * SPEED_OF_LIGHT / frequency;
}

float frequency_from_observation_time_seconds(float observation_time_seconds, int frequency_index){
    return frequency_index / observation_time_seconds;
}

float period_ms_from_frequency(float frequency){
    return 1000.0 / frequency;
}

double log_asymtotic_incomplete_gamma(double a, double z)
/*
  log_asymtotic_incomplete_gamma(double a, double z):
      Return the natural log of the incomplete gamma function in
          its asymtotic limit as z->infty.  This is from Abramowitz
          and Stegun eqn 6.5.32.
*/
{
    double x = 1.0, newxpart = 1.0, term = 1.0;
    int ii = 1;

    //printf("log_asymtotic_incomplete_gamma() being called with arguments:\n");
    //printf("   a = %f, z = %f\n", a, z);

    while (fabs(newxpart) > 1e-15) {
        term *= (a - ii);
        newxpart = term / pow(z, ii);
        x += newxpart;
        ii += 1;
        //printf("ii = %d, x = %f, newxpart = %f\n", ii, x, newxpart);
    }
    //printf("Took %d iterations.\n", ii);
    return (a - 1.0) * log(z) - z + log(x);
}

double log_asymtotic_gamma(double z)
/*
  log_asymtotic_gamma(double z):
      Return the natural log of the gamma function in its asymtotic limit
          as z->infty.  This is from Abramowitz and Stegun eqn 6.1.41.
*/
{
    double x, y;
    //printf("log_asymtotic_gamma() being called with argument z = %f\n", z);
    x = (z - 0.5) * log(z) - z + 0.91893853320467267;
    y = 1.0 / (z * z);
    x += (((-5.9523809523809529e-4 * y
            + 7.9365079365079365079365e-4) * y
           - 2.7777777777777777777778e-3) * y + 8.3333333333333333333333e-2) / z;
    return x;
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


double chi2_logp(double chi2, double dof)
/* MODIFIED FOR PULSCAN TO CLOSE INVALID REGION AT HIGH DOF */
/* Return the natural log probability corresponding to a chi^2 value */
/* of chi2 given dof degrees of freedom. */
{
    double logp;
    //printf("chi2 = %f, dof = %f\n", chi2, dof);

    if (chi2 <= 0.0) {
        return -INFINITY;
    }
    //printf("chi2/dof = %f\n", chi2/dof);
    // COMMENT OUT NEXT LINE IS THE MODIFICATION
    //if (chi2 / dof > 15.0 || (dof > 150 && chi2 / dof > 6.0)) {
    if (chi2 / dof > 1.5) {
        //printf("chi2/dof > 1.0\n");
        // printf("Using asymtotic expansion...\n");
        // Use some asymtotic expansions for the chi^2 distribution
        //   this is eqn 26.4.19 of A & S
        logp = log_asymtotic_incomplete_gamma(0.5 * dof, 0.5 * chi2) -
            log_asymtotic_gamma(0.5 * dof);
    } else {
        //printf("chi2/dof <= 1.0\n");
        int which, status;
        double p, q, bound, df = dof, x = chi2;

        which = 1;
        status = 0;
        // Determine the basic probability
        cdfchi(&which, &p, &q, &x, &df, &status, &bound);
        if (status) {
            printf("\nError in cdfchi() (chi2_logp()):\n");
            printf("   status = %d, bound = %g\n", status, bound);
            printf("   p = %g, q = %g, x = %g, df = %g\n\n", p, q, x, df);
            exit(1);
        }
        // printf("p = %.3g  q = %.3g\n", p, q);
        logp = log(q);
    }
    return logp;
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

double candidate_sigma(double power, int numsum, double numtrials)
/* Return the approximate significance in Gaussian       */
/* sigmas of a candidate of numsum summed powers,        */
/* taking into account the number of independent trials. */
{
    //printf("candidate_sigma() being called with arguments:\n");
    //printf("   power = %f, numsum = %d, numtrials = %f\n",
    //       power, numsum, numtrials);
    double logp, chi2, dof;

    if (power <= 0.0) {
        return 0.0;
    }

    // Get the natural log probability
    chi2 = 2.0 * power;
    dof = 2.0 * numsum;
    //printf("chi2 = %f, dof = %f\n", chi2, dof);
    logp = chi2_logp(chi2, dof);

    // Correct for numtrials
    logp += log(numtrials);

    // Convert to sigma
    return equivalent_gaussian_sigma(logp);
}

/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipies in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 */

/* Fast computation of the median of an array. */
/* Note:  It messes up the order!              */

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float median_function(float arr[], int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low)        /* One element only */
            return arr[median];

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

#undef ELEM_SWAP

void normalize_chunk_quickselect(float* chunk, size_t chunk_size) {
    if (chunk_size == 0) return;

    // Allocate memory for a copy of the chunk
    float* sorted_chunk = (float*) malloc(sizeof(float) * chunk_size);

    // Copy the chunk to sorted_chunk
    memcpy(sorted_chunk, chunk, sizeof(float) * chunk_size);

    // Compute the median using the new function
    float median = median_function(sorted_chunk, chunk_size);
    //printf("Median: %f\n", median);

    // Compute the MAD
    for (size_t i = 0; i < chunk_size; i++) {
        sorted_chunk[i] = fabs(sorted_chunk[i] - median); // Calculate the absolute deviation from the median
    }

    // Re-compute the median of the deviations to get the MAD
    float mad = median_function(sorted_chunk, chunk_size);
    //printf("MAD: %f\n", mad);

    // Free the allocated memory
    free(sorted_chunk);

    // Scale the MAD by the constant scale factor k
    float k = 1.4826f; // Scale factor to convert MAD to standard deviation for a normal distribution
    mad *= k;

    // Normalize the chunk
    if (mad != 0) {
        for (size_t i = 0; i < chunk_size; i++) {
            chunk[i] = (chunk[i] - median) / mad;
        }
    }
}

// function to compare floats for qsort
int compare_floats_median(const void *a, const void *b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;

    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

float* compute_magnitude_chunk_normalization_mad(const char *filepath, int *magnitude_size, int ncpus, int zmax) {
    // begin timer for reading input file
    double start = omp_get_wtime();
    size_t chunk_size = zmax * 30; // needs to be much larger than max boxcar width

    //printf("Reading file: %s\n", filepath);

    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Determine the size of the file
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t num_floats = filesize / sizeof(float);

    // Allocate memory for the data
    float* data = (float*) malloc(sizeof(float) * num_floats);
    if(data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    size_t n = fread(data, sizeof(float), num_floats, f);
    if (n % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        free(data);
        return NULL;
    }

    size_t size = n / 2;
    float* magnitude = (float*) malloc(sizeof(float) * size);
    if(magnitude == NULL) {
        printf("Memory allocation failed\n");
        free(data);
        return NULL;
    }

    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Reading the data took      %f seconds using 1 thread\n", time_spent);

    start = omp_get_wtime();

    #pragma omp parallel for
    // Perform chunk normalization
    for (size_t chunk_start = 0; chunk_start < size; chunk_start += chunk_size) {
        size_t chunk_end = chunk_start + chunk_size < size ? chunk_start + chunk_size : size;
        size_t current_chunk_size = chunk_end - chunk_start;

        // Separate the real and imaginary parts
        float* real_chunk = (float*) malloc(sizeof(float) * current_chunk_size);
        float* imag_chunk = (float*) malloc(sizeof(float) * current_chunk_size);

        if (real_chunk == NULL || imag_chunk == NULL) {
            printf("Memory allocation failed for real_chunk or imag_chunk\n");
            free(real_chunk);
            free(imag_chunk);
        }

        for (size_t i = 0; i < current_chunk_size; i++) {
            real_chunk[i] = data[2 * (chunk_start + i)];
            imag_chunk[i] = data[2 * (chunk_start + i) + 1];
        }

        // Normalize real and imaginary parts independently
        normalize_chunk_quickselect(real_chunk, current_chunk_size);
        normalize_chunk_quickselect(imag_chunk, current_chunk_size);

        // Recompute the magnitudes after normalization
        for (size_t i = chunk_start; i < chunk_end; i++) {
            magnitude[i] = real_chunk[i - chunk_start] * real_chunk[i - chunk_start] +
                        imag_chunk[i - chunk_start] * imag_chunk[i - chunk_start];
        }

        free(real_chunk);
        free(imag_chunk);
    }

    magnitude[0] = 0.0f; // set DC component of magnitude spectrum to 0

    fclose(f);
    free(data);

    *magnitude_size = (int) size;

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Normalizing the data took  %f seconds using %d thread(s)\n", time_spent, ncpus);
    return magnitude;
}

void decimate_array_2(float* input_array, float* output_array, int input_array_length){
    for (int i = 0; i < input_array_length/2; i++){
        output_array[i] = input_array[2*i] + input_array[2*i+1];
    }
}

void decimate_array_3(float* input_array, float* output_array, int input_array_length){
    for (int i = 0; i < input_array_length/3; i++){
        output_array[i] = input_array[3*i] + input_array[3*i+1] + input_array[3*i+2];
    }
}

// could reimplement this using a call of decimate array 2 on the array already decimated by 2
void decimate_array_4(float* input_array, float* output_array, int input_array_length){
    for (int i = 0; i < input_array_length/4; i++){
        output_array[i] = input_array[4*i] + input_array[4*i+1] + input_array[4*i+2] + input_array[4*i+3];
    }
}

__global__ void boxcarFilterKernel(float* magnitudesArray, candidate_struct_short* candidatesArray, int chunkSize, int zmax, int zstep, int nharmonics){
    // Allocate dynamic shared memory
    extern __shared__ char shared_memory[];

    int lookupArraySize = (chunkSize + zmax) * sizeof(float);
    int sumArraySize = chunkSize * sizeof(float);
    //int searchReduceArraySize = blockDim.x * sizeof(power_index_struct);

    // Calculate starting points using the char pointer
    float* lookupArray = (float*)shared_memory;
    float* sumArray = (float*)&shared_memory[lookupArraySize];
    power_index_struct* searchReduceArray = (power_index_struct*)&shared_memory[lookupArraySize + sumArraySize];

    int localIndex;
    int globalIndex;

    int numIterationsToFillLookupArray = (chunkSize + zmax + blockDim.x - 1)/blockDim.x;

    // fill lookuparray from magnitudesArray
    for (int i = 0; i < numIterationsToFillLookupArray;  i++){
        localIndex = threadIdx.x + i * blockDim.x;
        globalIndex = blockIdx.x * chunkSize + localIndex;
        if (localIndex < chunkSize + zmax){
            lookupArray[localIndex] = magnitudesArray[globalIndex];
        }
    }

    int numIterationsToCoverChunk = (chunkSize + blockDim.x - 1)/blockDim.x;
    
    // set sumArray to zero
    for (int i = 0; i < numIterationsToCoverChunk; i++){
        localIndex = threadIdx.x + i * blockDim.x;
        if (localIndex < chunkSize){
            sumArray[localIndex] = 0.0;
        }
    }

    float localMaxPower = 0;
    int localMaxIndex;
    int outputIndex = 0;


    for (int z = 0; z < zmax; z++){
        for (int i = 0; i < numIterationsToCoverChunk; i ++){
            localIndex = threadIdx.x + i * blockDim.x;
            if (localIndex < chunkSize){
                sumArray[localIndex] += lookupArray[localIndex + z];
            }
        }

        __syncthreads();

        if (z % zstep == 0){
            localMaxPower = 0;
            localMaxIndex = 0;
            for (int i = 0; i < numIterationsToCoverChunk; i++){
                localIndex = threadIdx.x + i * blockDim.x;
                if (localIndex < chunkSize){
                    if (sumArray[localIndex] > localMaxPower){
                        localMaxPower = sumArray[localIndex];
                        localMaxIndex = chunkSize * blockIdx.x + localIndex;
                    }
                }
            }
            searchReduceArray[threadIdx.x].power = localMaxPower;
            searchReduceArray[threadIdx.x].index = localMaxIndex;

            __syncthreads();

            for (int stride = blockDim.x/2; stride > 0; stride /= 2){
                if (threadIdx.x < stride){
                    if (searchReduceArray[threadIdx.x].power < searchReduceArray[threadIdx.x + stride].power){
                        searchReduceArray[threadIdx.x] = searchReduceArray[threadIdx.x + stride];
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0){
                candidatesArray[blockIdx.x*(zmax/zstep) + outputIndex].power = searchReduceArray[0].power;
                candidatesArray[blockIdx.x*(zmax/zstep) + outputIndex].index = searchReduceArray[0].index;
                candidatesArray[blockIdx.x*(zmax/zstep) + outputIndex].z = z;
                candidatesArray[blockIdx.x*(zmax/zstep) + outputIndex].harmonic = nharmonics;
            }
            outputIndex++;
        }
    }

    __syncthreads();



}

void recursive_boxcar_filter_cache_optimised(float* input_magnitudes_array, int magnitudes_array_length, \
                                int zmax, const char *filename, 
                                float observation_time_seconds, float sigma_threshold, int z_step, \
                                int chunkwidth, int ncpus, int nharmonics, int max_harmonics,
                                candidate_struct* global_candidates, int* global_candidates_array_index, \
                                int threads_per_block) {

    // make a copy of the input magnitudes array
    float* magnitudes_array = (float*)malloc(sizeof(float) * magnitudes_array_length);
    memcpy(magnitudes_array, input_magnitudes_array, sizeof(float) * magnitudes_array_length);

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s_ZMAX_%d_NUMHARM_%d_GPU.pulscand", base_name, zmax, max_harmonics);
    
    // open the file for writing.
    FILE *text_candidates_file = fopen(text_filename, "a");
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return;
    }
    
    // begin timer for decimation step
    double start_decimation = omp_get_wtime();

    float* magnitudes_array_decimated_sum;

    printf("------------- Beginning Search -------------\n");
    if (nharmonics == 1){
        //do nothing
    } else if (nharmonics == 2){
        magnitudes_array_decimated_sum = (float*) malloc(sizeof(float) * magnitudes_array_length/2);

        // make a copy of the magnitudes array, decimated by a factor of 2
        float* magnitudes_array_decimated_2 = (float*) malloc(sizeof(float) * magnitudes_array_length/2);
        decimate_array_2(magnitudes_array, magnitudes_array_decimated_2, magnitudes_array_length);

        // sum the first magnitudes_array_length/2 elements of the original + decimated arrays into a new array
        for (int i = 0; i < magnitudes_array_length/2; i++){
            magnitudes_array_decimated_sum[i] = magnitudes_array[i] + magnitudes_array_decimated_2[i];
        }

        magnitudes_array = magnitudes_array_decimated_sum;
        magnitudes_array_length /= 2;

        // end timer for decimation step
        double end_decimation = omp_get_wtime();
        double time_spent_decimation = end_decimation - start_decimation;
        printf("Decimation (2x) took       %f seconds using 1 thread\n", time_spent_decimation);

    } else if (nharmonics == 3){
        magnitudes_array_decimated_sum = (float*) malloc(sizeof(float) * magnitudes_array_length/3);
        
        // make a copy of the magnitudes array, decimated by a factor of 2
        float* magnitudes_array_decimated_2 = (float*) malloc(sizeof(float) * magnitudes_array_length/2);
        decimate_array_2(magnitudes_array, magnitudes_array_decimated_2, magnitudes_array_length);

        // make a copy of the magnitudes array, decimated by a factor of 3
        float* magnitudes_array_decimated_3 = (float*) malloc(sizeof(float) * magnitudes_array_length/3);
        decimate_array_3(magnitudes_array, magnitudes_array_decimated_3, magnitudes_array_length);

        // sum the first magnitudes_array_length/3 elements of the original + decimated arrays into a new array
        
        for (int i = 0; i < magnitudes_array_length/3; i++){
            magnitudes_array_decimated_sum[i] = magnitudes_array[i] + magnitudes_array_decimated_2[i] + magnitudes_array_decimated_3[i];
        }

        magnitudes_array = magnitudes_array_decimated_sum;
        magnitudes_array_length /= 3;

        // end timer for decimation step
        double end_decimation = omp_get_wtime();
        double time_spent_decimation = end_decimation - start_decimation;
        printf("Decimation (3x) took       %f seconds using 1 thread\n", time_spent_decimation);
    } else if (nharmonics == 4){
        magnitudes_array_decimated_sum = (float*) malloc(sizeof(float) * magnitudes_array_length/4);
        
        // make a copy of the magnitudes array, decimated by a factor of 2
        float* magnitudes_array_decimated_2 = (float*) malloc(sizeof(float) * magnitudes_array_length/2);
        decimate_array_2(magnitudes_array, magnitudes_array_decimated_2, magnitudes_array_length);

        // make a copy of the magnitudes array, decimated by a factor of 3
        float* magnitudes_array_decimated_3 = (float*) malloc(sizeof(float) * magnitudes_array_length/3);
        decimate_array_3(magnitudes_array, magnitudes_array_decimated_3, magnitudes_array_length);

        // make a copy of the magnitudes array, decimated by a factor of 4
        float* magnitudes_array_decimated_4 = (float*) malloc(sizeof(float) * magnitudes_array_length/4);
        decimate_array_4(magnitudes_array, magnitudes_array_decimated_4, magnitudes_array_length);

        // sum the first magnitudes_array_length/4 elements of the original + decimated arrays into a new array
        
        for (int i = 0; i < magnitudes_array_length/4; i++){
            magnitudes_array_decimated_sum[i] = magnitudes_array[i] + magnitudes_array_decimated_2[i] + magnitudes_array_decimated_3[i] + magnitudes_array_decimated_4[i];
        }

        magnitudes_array = magnitudes_array_decimated_sum;
        magnitudes_array_length /= 4;

        // end timer for decimation step
        double end_decimation = omp_get_wtime();
        double time_spent_decimation = end_decimation - start_decimation;
        printf("Decimation (4x) took       %f seconds using 1 thread\n", time_spent_decimation);
    } else {
        printf("ERROR: nharmonics must be 1, 2, 3 or 4\n");
        return;
    }

    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array++;
    magnitudes_array_length--;

    int valid_length = magnitudes_array_length;
    int initial_length = magnitudes_array_length;

    double num_independent_trials = ((double)zmax)*((double)initial_length)/6.95; // 6.95 from eqn 6 in Anderson & Ransom 2018

    int num_candidates_per_chunk = zmax / z_step;

    int num_chunks = (valid_length + chunkwidth - 1) / chunkwidth;
    
    // begin timer for boxcar filtering
    double start = omp_get_wtime();

    // allocate memory on GPU for magnitudes array
    float* device_magnitudes_array;
    cudaMalloc(&device_magnitudes_array, sizeof(float) * magnitudes_array_length);
    cudaMemcpy(device_magnitudes_array, magnitudes_array, sizeof(float) * magnitudes_array_length, cudaMemcpyHostToDevice);

    // allocate memory on GPU for candidates array
    candidate_struct_short* device_candidates;
    cudaMalloc(&device_candidates, sizeof(candidate_struct_short) * num_chunks * num_candidates_per_chunk);

    int numThreadsPerBlock = threads_per_block;
    int numBlocks = num_chunks;

    size_t lookupArraySize = (chunkwidth + zmax) * sizeof(float);
    size_t sumArraySize = chunkwidth * sizeof(float);
    size_t searchReduceArraySize = numThreadsPerBlock * sizeof(power_index_struct);

    size_t shared_memory_size = lookupArraySize + sumArraySize + searchReduceArraySize;

    printf("Required GPU smem          %ld bytes\n", shared_memory_size);

    cudaFuncSetAttribute(boxcarFilterKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

    // check it worked by getting last cuda error
    cudaError_t smemError = cudaGetLastError();
    if (smemError != cudaSuccess) {
        printf("CUDA error at smem request (probably requested too much shared memory, reduce zmax or chunkwidth): %s\n", cudaGetErrorString(smemError));
        // exit the program
        exit(1);
    }

    // Check the current device properties
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    // print all arguments before calling kernel
    //printf("numBlocks = %d, numThreadsPerBlock = %d, shared_memory_size = %ld\n", numBlocks, numThreadsPerBlock, shared_memory_size);
    boxcarFilterKernel<<<numBlocks, numThreadsPerBlock, shared_memory_size>>>(device_magnitudes_array, device_candidates, chunkwidth, zmax, z_step, nharmonics);
    // get last cuda error
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    cudaDeviceSynchronize();



    // copy candidates to host and print
    candidate_struct_short* hostCopyCandidates;
    hostCopyCandidates = (candidate_struct_short*) malloc(sizeof(candidate_struct_short) * num_chunks * num_candidates_per_chunk);
    cudaMemcpy(hostCopyCandidates, device_candidates, sizeof(candidate_struct_short) * num_chunks * num_candidates_per_chunk, cudaMemcpyDeviceToHost);


    // end timer for boxcar filtering
    double end = omp_get_wtime();

    double time_spent = end - start;
    printf("Searching the data took    %f seconds using %s\n", time_spent, props.name);

    start = omp_get_wtime();
    int degrees_of_freedom = 1;
    if (nharmonics == 1){
        degrees_of_freedom  = 1;
    } else if (nharmonics == 2){
        degrees_of_freedom  = 3;
    } else if (nharmonics == 3){
        degrees_of_freedom  = 6;
    } else if (nharmonics == 4){
        degrees_of_freedom  = 10;
    }

    double* sigma_array;
    sigma_array = (double*) malloc(sizeof(double) * num_chunks * num_candidates_per_chunk);

    for (int i = 0; i < num_chunks * num_candidates_per_chunk; i++){
        sigma_array[i] = candidate_sigma(hostCopyCandidates[i].power*0.5, (hostCopyCandidates[i].z+1)*degrees_of_freedom, num_independent_trials);
    }
    
    end = omp_get_wtime();
    time_spent = end - start;
    printf("Candidate sigma took       %f seconds using 1 thread\n", time_spent);
    start = omp_get_wtime();

    for (int i = 0; i < num_chunks * num_candidates_per_chunk; i++){
        if (sigma_array[i] > sigma_threshold){
            global_candidates[*global_candidates_array_index].power = hostCopyCandidates[i].power;
            global_candidates[*global_candidates_array_index].index = hostCopyCandidates[i].index;
            global_candidates[*global_candidates_array_index].z = hostCopyCandidates[i].z;
            global_candidates[*global_candidates_array_index].harmonic = hostCopyCandidates[i].harmonic;
            global_candidates[*global_candidates_array_index].sigma = sigma_array[i];
            *global_candidates_array_index = *global_candidates_array_index + 1;
        }
    }

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Applying threshold took    %f seconds using 1 thread\n", time_spent);
    start = omp_get_wtime();
    fclose(text_candidates_file);
    free(base_name);
    free(hostCopyCandidates);
    //free(final_output_candidates);

}

const char* pulscan_frame = 
"    .          .     .     *        .   .   .     .\n"
"         " BOLD "___________      . __" RESET " .  .   *  .   .  .  .     .\n"
"    . *   " BOLD "_____  __ \\__+ __/ /_____________ _____" RESET " .    " FLASHING "*" RESET "  .\n"
"  +    .   " BOLD "___  /_/ / / / / / ___/ ___/ __ `/ __ \\" RESET "     + .\n"
" .          " BOLD "_  ____/ /_/ / (__  ) /__/ /_/ / / / /" RESET " .  *     . \n"
"       .    " BOLD "/_/ *  \\__,_/_/____/\\___/\\__,_/_/ /_/" RESET "    \n"
"    *    +     .     .     . +     .     +   .      *   +\n"

"  J. White, K. Adámek, J. Roy, S. Ransom, W. Armour  2023\n\n";


int main(int argc, char *argv[]) {
    cudaDeviceSynchronize();
    // start overall program timer
    double start_program = omp_get_wtime();
    printf("%s\n", pulscan_frame);

    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-numharm int] [-tobs float] [-sigma float] [-zstep int] [-chunkwidth int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\t\tThe input file path (.fft file format such as the output of realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-ncpus [int]\t\tThe number of OpenMP threads to use (default 1)\n");
        printf("\t-zmax [int]\t\tThe max boxcar width (default = 200, similar meaning to zmax in accelsearch)\n");
        printf("\t-numharm [int]\t\tThe maximum number of harmonics to sum (default = 1, options are 1, 2, 3 or 4)\n");
        printf("\t-tobs [float]\t\tThe observation time in seconds, this " BOLD "MUST BE SPECIFIED ACCURATELY" RESET " if you want physical frequency/acceleration values.\n");
        printf("\t-sigma [float]\t\tThe sigma threshold (default = 2.0), candidates with sigma below this value will not be written to the output file\n");
        printf("\t-zstep [int]\t\tThe step size in z (default = 2).\n");
        printf("\t-chunkwidth [int]\tThe chunk width (units are r-bins, default = 32768), you will get up to ( rmax * zmax ) / ( chunkwidth * zstep ) candidates\n");
        printf("\t-threadsperblock [int]\tThe number of threads per block (default = 256)\n");
        return 1;
    }

    // Get the number of OpenMP threads from the command line arguments
    // If not provided, default to 1
    int ncpus = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-ncpus") == 0 && i+1 < argc) {
            ncpus = atoi(argv[i+1]);
        }
    }

    // Get the zmax from the command line arguments
    // If not provided, default to 200
    int zmax = 200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            zmax = atoi(argv[i+1]);
        }
    }

    // Get the observation time from the command line arguments
    // If not provided, default to 0.0
    float observation_time_seconds = 0.0f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-tobs") == 0 && i+1 < argc) {
            observation_time_seconds = atof(argv[i+1]);
        }
    }

    if (observation_time_seconds == 0.0f) {
        printf(RED FLASHING "WARNING" RESET ": No observation time provided, frequency and acceleration values will be inaccurate.\n");
        printf("[Optional] Please specify an observation time with the -tobs flag, e.g. -tobs 600.0\n\n");
    }

    // Get the number of harmonics to sum from the command line arguments
    // If not provided, default to 1
    int nharmonics = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-numharm") == 0 && i+1 < argc) {
            nharmonics = atoi(argv[i+1]);
        }
    }

    // Get the sigma threshold value from the command line arguments
    // If not provided, default to 2.0
    float sigma_threshold = 2.0f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-sigma") == 0 && i+1 < argc) {
            sigma_threshold = atof(argv[i+1]);
        }
    }

    // Get the z step size from the command line arguments
    // If not provided, default to 2
    int z_step = 2;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zstep") == 0 && i+1 < argc) {
            z_step = atoi(argv[i+1]);
        }
    }

    // Get the chunk width from the command line arguments
    // If not provided, default to 4096
    int chunkwidth = 4096;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-chunkwidth") == 0 && i+1 < argc) {
            chunkwidth = atoi(argv[i+1]);
        }
    }

    // Get the number of threads per block from the command line arguments
    // If not provided, default to 256
    int threads_per_block = 256;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-threadsperblock") == 0 && i+1 < argc) {
            threads_per_block = atoi(argv[i+1]);
        }
    }

    omp_set_num_threads(ncpus);

    int magnitude_array_size;
    float* magnitudes = compute_magnitude_chunk_normalization_mad(argv[1], &magnitude_array_size, ncpus, zmax);

    if(magnitudes == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    // Extract file name without extension
    char *base_name = strdup(argv[1]);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char text_filename[255];
    snprintf(text_filename, 255, "%s_ZMAX_%d_NUMHARM_%d_GPU.pulscand", base_name, zmax, nharmonics);

    FILE *text_candidates_file = fopen(text_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (text_candidates_file == NULL) {
        printf("Could not open file for writing text results.\n");
        return 1;
    }

    fprintf(text_candidates_file, "%10s %10s %10s %14s %10s %14s %10s %14s %14s %10s\n", 
        "sigma", "power", "period_ms", "frequency_hz", "rbin", "f-dot", "z", "acceleration", "logp", "harmonic");
    
    

    int num_chunks = (magnitude_array_size + chunkwidth - 1) / chunkwidth;

    int max_candidates_per_harmonic = zmax*num_chunks;
    candidate_struct *global_candidates_array = (candidate_struct*) malloc(sizeof(candidate_struct) * nharmonics * max_candidates_per_harmonic);
    int global_candidates_array_index = 0;

    for (int harmonic = 1; harmonic <= nharmonics; harmonic++){
        recursive_boxcar_filter_cache_optimised(magnitudes, 
            magnitude_array_size, 
            zmax, 
            argv[1],
            observation_time_seconds, 
            sigma_threshold,
            z_step,
            chunkwidth,
            ncpus,
            harmonic,
            nharmonics,
            global_candidates_array,
            &global_candidates_array_index,
            threads_per_block);
    }

    // begin timer for sorting output array
    double start_sort = omp_get_wtime();

    int num_candidates = global_candidates_array_index;
    qsort(global_candidates_array, num_candidates, sizeof(candidate_struct), compare_candidate_structs_sigma);
    // end timer for sorting output array
    double end_sort = omp_get_wtime();
    double time_spent_sort = end_sort - start_sort;
    printf("Sorting candidates took    %f seconds using 1 thread\n", time_spent_sort);

    // start timer for writing final_output_candidates to text file
    double start_write = omp_get_wtime();

    float temp_period_ms;
    float temp_frequency;
    float temp_fdot;
    float temp_acceleration;
    float temp_logp;

    // write final_output_candidates to text file with physical measurements
    for (int i = 0; i < num_candidates; i++){
        if (global_candidates_array[i].sigma > sigma_threshold){
            //if (global_candidates_array[i].index > 0){
                temp_period_ms = period_ms_from_frequency(frequency_from_observation_time_seconds(observation_time_seconds,global_candidates_array[i].index));
                temp_frequency = frequency_from_observation_time_seconds(observation_time_seconds,global_candidates_array[i].index);
                temp_fdot = fdot_from_boxcar_width(global_candidates_array[i].z, observation_time_seconds);
                temp_acceleration = acceleration_from_fdot(fdot_from_boxcar_width(global_candidates_array[i].z, observation_time_seconds), frequency_from_observation_time_seconds(observation_time_seconds,global_candidates_array[i].index));
                int degrees_of_freedom = 1;
                if (global_candidates_array[i].harmonic == 1){
                    degrees_of_freedom  = 1;
                } else if (global_candidates_array[i].harmonic == 2){
                    degrees_of_freedom  = 3;
                } else if (global_candidates_array[i].harmonic == 3){
                    degrees_of_freedom  = 6;
                } else if (global_candidates_array[i].harmonic == 4){
                    degrees_of_freedom  = 10;
                } else {
                    printf("ERROR: nharmonics must be 1, 2, 3 or 4\n");
                    return 1;
                }
                temp_logp = chi2_logp(global_candidates_array[i].power, degrees_of_freedom * global_candidates_array[i].z * 2);
                fprintf(text_candidates_file, "%10.6lf %10.4f %10.6f %14.6f %10ld %14.8f %10d %14.6f %14.6f %10d\n", 
                    global_candidates_array[i].sigma,
                    global_candidates_array[i].power,
                    temp_period_ms,
                    temp_frequency,
                    global_candidates_array[i].index,
                    temp_fdot,
                    global_candidates_array[i].z,
                    temp_acceleration,
                    temp_logp,
                    global_candidates_array[i].harmonic);
            //}
        }
    }

    //printf("global_candidates_array_index = %d\n", global_candidates_array_index);
    /*for (int i = 0; i < global_candidates_array_index; i++){
        if (global_candidates_array[i].sigma > sigma_threshold){
            //printf("sigma = %f, harmonic = %d\n", global_candidates_array[i].sigma, global_candidates_array[i].harmonic);
        }
    }*/
    fclose(text_candidates_file);

    // end timer for writing final_output_candidates to text file
    double end_write = omp_get_wtime();
    double time_spent_write = end_write - start_write;
    printf("Writing the output took    %f seconds using 1 thread\n", time_spent_write);


    free(magnitudes);

    // end overall program timer
    double end_program = omp_get_wtime();
    double time_spent_program = end_program - start_program;
    printf("--------------------------------------------\nTotal time spent was       " GREEN "%f seconds" RESET " (excluding GPU initialisation)\n\n\n", time_spent_program);

    printf("Output written to %s\n", text_filename);

    return 0;
}