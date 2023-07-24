compile with `gcc pulscan.c -o pulscan -lm -fopenmp -Ofast -ftree-vectorize`

run with `./pulscan`

post process the candidate file with `python3 make_formatted_candidate_list PATH_TO_CANDIDATE_FILE.bctxtcand