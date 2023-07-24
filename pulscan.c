// Jack White 2023, jack.white@eng.ox.ac.uk

// compile with "gcc pulscan.c -o pulscan -lm -fopenmp -Ofast -ftree-vectorize -ffast-math -fopt-info-vec-optimized"

// run with "./pulscan"

// This program reads in a .fft file produced by PRESTO realfft
// and computes the boxcar filter candidates for a range of boxcar widths, 1 to zmax (default 1200)
// The number of candidates per boxcar is set by the user, default 10

// The output is a text file called INPUTFILENAME.bctxtcand with the following columns:
// boxcar_width,frequency_bin_index,power

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_DATA_SIZE 10000000 // assume file won't be larger than this, 10M samples, increase if required
#define DEFAULT_CANDIDATES_PER_BOXCAR 10

typedef struct {
    int boxcar_width;
    int frequency_index;
    float power;
} Candidate;

float* compute_magnitude(const char *filepath, int *magnitude_size) {
    printf("Reading file: %s\n", filepath);

    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    float* data = (float*) malloc(sizeof(float) * MAX_DATA_SIZE);
    if(data == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }
    size_t n = fread(data, sizeof(float), MAX_DATA_SIZE, f);
    if (n % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        free(data);
        return NULL;
    }

    // compute mean and variance of real and imaginary components, ignoring DC component

    float real_sum = 0.0, imag_sum = 0.0;
    for(int i = 1; i < n / 2; i++) {
        real_sum += data[2 * i];
        imag_sum += data[2 * i + 1];
    }
    float real_mean = real_sum / ((n-1) / 2);
    float imag_mean = imag_sum / ((n-1) / 2);

    float real_variance = 0.0, imag_variance = 0.0;
    for(int i = 1; i < n / 2; i++) {
        real_variance += pow((data[2 * i] - real_mean), 2);
        imag_variance += pow((data[2 * i + 1] - imag_mean), 2);
    }
    real_variance /= ((n-1) / 2);
    imag_variance /= ((n-1) / 2);

    float real_stdev = sqrt(real_variance);
    float imag_stdev = sqrt(imag_variance);

    float* magnitude = (float*) malloc(sizeof(float) * n / 2);
    if(magnitude == NULL) {
        printf("Memory allocation failed\n");
        free(data);
        return NULL;
    }

    // set DC component of magnitude spectrum to 0
    magnitude[0] = 0.0f;

    for (int i = 1; i < n / 2; i++) {
        float norm_real = (data[2 * i] - real_mean) / real_stdev;
        float norm_imag = (data[2 * i + 1] - imag_mean) / imag_stdev;
        magnitude[i] = pow(norm_real, 2) + pow(norm_imag, 2);
    }

    fclose(f);
    free(data);

    // pass the size of the magnitude array back through the output parameter
    *magnitude_size = n / 2;

    // return the pointer to the magnitude array
    return magnitude;
}


void recursive_boxcar_filter(float* magnitudes_array, int magnitudes_array_length, int max_boxcar_width, const char *filename, int candidates_per_boxcar) {
    printf("Computing boxcar filter candidates for %d boxcar widths...\n", max_boxcar_width);

    // Extract file name without extension
    char *base_name = strdup(filename);
    char *dot = strrchr(base_name, '.');
    if(dot) *dot = '\0';

    // Create new filename
    char new_filename[255];
    snprintf(new_filename, 255, "%s.bctxtcand", base_name);
    printf("Storing %d candidates per boxcar in text format in %s\n", candidates_per_boxcar, new_filename);
    
    FILE *candidates = fopen(new_filename, "w"); // open the file for writing. Make sure you have write access in this directory.
    if (candidates == NULL) {
        printf("Could not open file for writing results.\n");
        return;
    }
    fprintf(candidates, "boxcar_width,frequency_bin_index,power\n");


    // we want to ignore the DC component, so we start at index 1, by adding 1 to the pointer
    magnitudes_array += 1;
    magnitudes_array_length -= 1;

    int valid_length = magnitudes_array_length;
    int offset = 0;

    float* temp_sum_array = (float*) malloc(sizeof(float) * magnitudes_array_length);
    memcpy(temp_sum_array, magnitudes_array, sizeof(float) * magnitudes_array_length);

    for (int boxcar_width = 2; boxcar_width < max_boxcar_width; boxcar_width++) {
        valid_length -= 1;
        offset += 1;

        float current_max = 0;
        #pragma omp parallel for
        for (int i = 0; i < valid_length; i++) {
            temp_sum_array[i] += magnitudes_array[i + offset];
        }

        Candidate top_candidates[candidates_per_boxcar];

        int window_length = valid_length / candidates_per_boxcar;
        
        #pragma omp parallel for
        for (int i = 0; i < candidates_per_boxcar; i++) {
            float local_max_power = -INFINITY;
            int window_start = i * window_length;
            for (int j = window_start; j < window_start + window_length; j++){
                if (temp_sum_array[j] > local_max_power) {
                    local_max_power = temp_sum_array[j];
                    top_candidates[i].frequency_index = j;
                    top_candidates[i].power = local_max_power;
                    top_candidates[i].boxcar_width = boxcar_width;
                }
            }
            fprintf(candidates, "%d,%d,%f\n", top_candidates[i].boxcar_width, top_candidates[i].frequency_index, top_candidates[i].power);
        }
    }

    free(temp_sum_array);
    fclose(candidates);
    free(base_name);
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-candidates int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-ncpus [int]\tThe number of OpenMP threads to use (default 1)\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 1200, max = the size of your input data)\n");
        printf("\t-candidates [int]\tThe number of candidates per boxcar (default = 10), total candidates in output will be = zmax * candidates\n");
        return 1;
    }

    // Get the number of candidates per boxcar from the command line arguments
    // If not provided, default to 10
    int candidates_per_boxcar = DEFAULT_CANDIDATES_PER_BOXCAR;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-candidates") == 0 && i+1 < argc) {
            candidates_per_boxcar = atoi(argv[i+1]);
        }
    }

    // Get the number of OpenMP threads from the command line arguments
    // If not provided, default to 1
    int num_threads = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-ncpus") == 0 && i+1 < argc) {
            num_threads = atoi(argv[i+1]);
        }
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 1200
    int max_boxcar_width = 1200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            max_boxcar_width = atoi(argv[i+1]);
        }
    }

    omp_set_num_threads(num_threads);

    int magnitude_array_size;
    float* magnitudes = compute_magnitude(argv[1], &magnitude_array_size);

    if(magnitudes == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    recursive_boxcar_filter(magnitudes, magnitude_array_size, max_boxcar_width, argv[1], candidates_per_boxcar);

    return 0;
}
