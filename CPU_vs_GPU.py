import subprocess
import re

# Function to run a command and parse the output
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return ""

# Function to extract times from the output
def parse_times(output):
    # Adjust the regex to skip the ANSI escape codes
    total_time = float(re.search(r"Total time spent was\s+\033\[32m([\d\.]+) seconds\033\[0m", output).group(1))
    search_times = re.findall(r"Searching the data took\s+([\d\.]+) seconds", output)
    search_time_sum = sum(map(float, search_times))
    return total_time, search_time_sum

# Constants
ncpus_values = [1, 2, 4, 8, 16]
num_repeats = 16
programs = ['pulscan', 'pulscan_hybrid']
base_command = "./{} sample_data/1298us_binary.fft -zmax 1000 -numharm 4 -chunkwidth 24000 -ncpus {}"

# Dictionary to store results
results = {program: {ncpus: {'total_times': [], 'search_sums': []} for ncpus in ncpus_values} for program in programs}

# Main execution loop
for program in programs:
    for ncpus in ncpus_values:
        for _ in range(num_repeats):
            command = base_command.format(program, ncpus)
            # Append the threadsperblock flag if the program is pulscan_hybrid
            if program == 'pulscan_hybrid':
                command += " -threadsperblock 512"
            output = run_command(command)
            total_time, search_sum = parse_times(output)
            results[program][ncpus]['total_times'].append(total_time)
            results[program][ncpus]['search_sums'].append(search_sum)

# Print results
for program in programs:
    print(f"Results for {program}:")
    for ncpus in ncpus_values:
        avg_total = sum(results[program][ncpus]['total_times']) / num_repeats
        avg_search_sum = sum(results[program][ncpus]['search_sums']) / num_repeats
        print(f" - ncpus={ncpus}: Average Total Time = {avg_total:.3f}s, Average Search Time Sum = {avg_search_sum:.3f}s")
    print()
