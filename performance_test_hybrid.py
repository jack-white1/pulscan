import subprocess
import re
import statistics
import sys

def run_command(command):
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout

def extract_execution_time(output):
    # Regular expression to find the execution time
    match = re.search(r'Total time spent was\s+\x1b\[32m([\d.]+) seconds\x1b\[0m', output)
    if match:
        return float(match.group(1))
    return None

def main():
    # Define ranges for the -ncpus and -zmax options
    ncpus_values = [1, 2, 4, 8, 16, 32, 64, 128]
    zmax_values = [10, 100, 1000]
    num_runs = 16  # Number of runs per configuration
    total_tests = len(ncpus_values) * len(zmax_values) * num_runs
    current_test = 1

    # Results storage
    results = {}

    # Iterate over all combinations of ncpus and zmax
    for ncpus in ncpus_values:
        for zmax in zmax_values:
            times = []
            # Process each configuration multiple times
            for _ in range(num_runs):
                sys.stdout.write(f"\rRunning test {current_test} of {total_tests}...")  # Update progress on the same line
                sys.stdout.flush()
                command = f"./pulscan_hybrid sample_data/1298us_binary.fft -chunkwidth 4096 -numharm 4 -tobs 602.112 -ncpus {ncpus} -zmax {zmax}"
                output = run_command(command)
                execution_time = extract_execution_time(output)
                if execution_time is not None:
                    times.append(execution_time)
                else:
                    print(f"\nFailed to extract execution time from output for ncpus={ncpus}, zmax={zmax}")
                current_test += 1

            # Calculate average and standard deviation
            if times:
                average_time = statistics.mean(times)
                stdev_time = statistics.stdev(times) if len(times) > 1 else 0
                results[(ncpus, zmax)] = (average_time, stdev_time)

    # Print results
    print("\n\nSummary of results:")
    for (ncpus, zmax), (avg, stdev) in results.items():
        print(f"Config (ncpus={ncpus}, zmax={zmax}): Average Time = {avg:.6f} s, StdDev = {stdev:.6f} s")

if __name__ == "__main__":
    main()
