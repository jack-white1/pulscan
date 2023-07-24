# PULSCAN

A PRESTO-compatible implementation of the boxcar acceleration search (in development) for detecting the signatures of binary pulsars in PRESTO .fft files.

**EXPECT LOWER SENSITIVITY THAN PRESTO'S FDAS ACCEL_SEARCH**

## INSTRUCTIONS
1. Compile with `gcc pulscan.c -o pulscan -lm -fopenmp -Ofast -ftree-vectorize`

2. Run with `./pulscan FILENAME -ncpus XXX -zmax XXX - candidates XXX`
    - REQUIRED argument `filename`
    - OPTIONAL argument `-ncpus XXX` The (integer) number of OpenMP threads to use (default 1)
    - OPTIONAL argument `-zmax XXX` The (integer) max boxcar width (default = 1200, max = the size of your input data)
    - OPTIONAL argument `-candidates XXX` The (integer) number of candidates per boxcar width to produce
    - The total number of candidates will be `zmax` * `candidates`

3. Post process the candidate file to make a human readable version with `python3 make_formatted_candidate_list.py PATH_TO_CANDIDATE_FILE.bctxtcand`

## EXAMPLE
0. Dedisperse your filterbank with PRESTO's `prepsubband` and make your .fft file using PRESTO's `realfft` command
1. `gcc pulscan.c -o pulscan -lm -fopenmp -Ofast -ftree-vectorize`
2. `./pulscan ./test_data/test.fft`
3. `python3 make_formatted_candidate_list.py ./test_data/test.bctxtcand`