import subprocess
import re
import csv
import platform
import math

def get_cpu_model():
    """Return the CPU model name formatted for a filename."""
    cpu_info = platform.processor()
    return cpu_info.replace(" ", "_").replace("@", "").replace(",", "")

def run_pulscan(boxcar_chunk_width, normalize_chunk_width, num_cpus):
    """Run the pulscan command with specified boxcar chunk widths and return the total time."""
    command = [
        "./pulscan", "sample_data/1298us_binary.fft",
        "-boxcarchunkwidth", str(boxcar_chunk_width),
        "-normalizechunkwidth", str(normalize_chunk_width),
        "-ncpus", str(num_cpus)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    time_search = re.search(r"Total time spent was\s+\033\[32m([0-9.]+) seconds", output)
    if time_search:
        return float(time_search.group(1))
    else:
        raise ValueError("Time output not found in the response")

def find_optimal_combinations(max_width, num_cpus):
    """Find the optimal combination of boxcar chunk width and normalize chunk width."""
    times = []
    boxcar_chunk_widths = normalize_chunk_widths = [256 * (2 ** i) for i in range(int(math.log(max_width / 256, 2)) + 1)]
    
    for boxcar_chunk_width in boxcar_chunk_widths:
        for normalize_chunk_width in boxcar_chunk_widths:
            run_times = []
            for i in range(16):  # Run the same configuration 16 times
                try:
                    time_spent = run_pulscan(boxcar_chunk_width, normalize_chunk_width, num_cpus)
                    run_times.append(time_spent)
                    print(f"Run {i+1}: Boxcar Chunk Width = {boxcar_chunk_width}, Normalize Chunk Width = {normalize_chunk_width}, Time = {time_spent}s")
                except ValueError as e:
                    print(f"Error: {e}")
                    run_times.append(None)

            if all(time is not None for time in run_times):
                average_time = sum(filter(None, run_times)) / len(run_times)  # Only average non-None values
            else:
                average_time = None
            times.append([boxcar_chunk_width, normalize_chunk_width] + run_times + [average_time])
            print(f"Averaged: Boxcar Chunk Width = {boxcar_chunk_width}, Normalize Chunk Width = {normalize_chunk_width}, Average Time = {average_time}s")
                
    return times

def save_times_data(times, num_cpus):
    """Save the times data to a CSV file named by the CPU model and num_cpus."""
    cpu_model = get_cpu_model()
    filename = f"raw_times_{cpu_model}_ncpus{num_cpus}.csv"
    num_runs = 16
    header = ["Boxcar Chunk Width", "Normalize Chunk Width"] + [f"Run {i+1}" for i in range(num_runs)] + ["Average Time"]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(times)
    print(f"Data saved to {filename}")

# Example usage
if __name__ == "__main__":
    max_width = 131072  # 128K
    num_cpus = 1
    times = find_optimal_combinations(max_width, num_cpus)
    save_times_data(times, num_cpus)
