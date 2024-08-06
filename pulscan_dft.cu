#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define RESET   "\033[0m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"
#define RED     "\033[31m"

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

    // Check filepath ends with ".dat"
    if (strlen(filepath) < 5 || strcmp(filepath + strlen(filepath) - 4, ".dat") != 0) {
        printf("Input file must be a .dat file.\n");
        return 1;
    }

    FILE *f = fopen(filepath, "rb");

    // Determine the size of the file in bytes
    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Read the file into CPU memory
    size_t numFloats = filesize / sizeof(float);

    printf("Filesize:                              %zu bytes\n", filesize);
    printf("Number of floats in file:              %zu\n", numFloats);

    // Cap the filesize at the nearest lower factor of 8192 for compatibility later on
    //numFloats = numFloats - (numFloats % 8192);
    float* rawData = (float*) malloc(sizeof(float) * numFloats);
    size_t itemsRead = fread(rawData, sizeof(float), numFloats, f);
    if (itemsRead != numFloats) {
        // Handle error: not all items were read
        fprintf(stderr, "Error reading file: only %zu out of %zu items read\n", itemsRead, numFloats);
        // You might want to take additional action here, like exiting the function or the program
    }
    fclose(f);

    // make a copy of the fft filename called inffilename
    char *inffilename = strdup(argv[1]);

    // replace the last 3 characters of the string (which should be "fft" with "inf")
    inffilename[strlen(inffilename) - 3] = 'i';
    inffilename[strlen(inffilename) - 2] = 'n';
    inffilename[strlen(inffilename) - 1] = 'f';

    // open the inf file and look for the line starting with
    // "Width of each time series bin (sec)    =  "
    FILE *inf_file = fopen(inffilename, "r");
    if (inf_file == NULL) {
        printf(RED FLASHING "ERROR" RESET ": Could not open %s file.\n", inffilename);
        printf("Please ensure that the .inf file is in the same directory as the .fft file.\n");
        free(inffilename);  // Free allocated memory
        return 1;
    }

    char *line = NULL;
    size_t len = 0;
    size_t read;
    float width_of_each_time_series_bin = 0.0f;
    while ((read = getline(&line, &len, inf_file)) != -1) {
        if (strstr(line, "Width of each time series bin (sec)    =  ") != NULL) {
            char *width_of_each_time_series_bin_str = line + 42;
            width_of_each_time_series_bin = atof(width_of_each_time_series_bin_str);
            break;
        }
    }

    printf("Width of each time series bin (sec):    %f\n", width_of_each_time_series_bin);

    free(line); // Free the line buffer

    fclose(inf_file); // Close the file

    if (width_of_each_time_series_bin == 0.0f) {
        printf(RED FLASHING "ERROR" RESET ": Could not find the width of each time series bin in the .inf file.\n");
        printf("Please ensure that the .inf file is in the correct format.\n");
        free(inffilename);  // Free allocated memory
        return 1;
    }

    // stop timing
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Reading file took:                      %f ms\n", (float)duration.count());

    // start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU processing took:                    %f ms\n", milliseconds);

    // check last cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;
}