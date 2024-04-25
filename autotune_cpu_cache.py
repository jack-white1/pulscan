import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import platform
import os

def get_cpu_model():
    """Return the CPU model name formatted for a filename."""
    cpu_info = platform.processor()
    # Remove problematic characters for filenames
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

def find_optimal_combinations(min_width, max_width, step, num_cpus):
    """Find the optimal combination of boxcar chunk width and normalize chunk width."""
    size = (max_width - min_width) // step + 1
    times = []
    boxcar_chunk_widths = normalize_chunk_widths = np.arange(min_width, max_width + 1, step)
    
    for boxcar_chunk_width in boxcar_chunk_widths:
        for normalize_chunk_width in normalize_chunk_widths:
            run_times = []
            for _ in range(16):  # Run the same configuration 16 times
                try:
                    time_spent = run_pulscan(boxcar_chunk_width, normalize_chunk_width, num_cpus)
                    run_times.append(time_spent)
                    print(f"Run {_+1}: Boxcar Chunk Width = {boxcar_chunk_width}, Normalize Chunk Width = {normalize_chunk_width}, Time = {time_spent}s")
                except ValueError as e:
                    print(f"Error: {e}")
                    run_times.append(np.nan)

            average_time = np.nanmean(run_times)
            times.append([boxcar_chunk_width, normalize_chunk_width] + run_times + [average_time])
            print(f"Averaged: Boxcar Chunk Width = {boxcar_chunk_width}, Normalize Chunk Width = {normalize_chunk_width}, Average Time = {average_time}s")
                
    return np.array(times)

def save_times_data(times, num_cpus):
    """Save the times data to a CSV file named by the CPU model and num_cpus."""
    cpu_model = get_cpu_model()
    filename = f"raw_times_{cpu_model}_ncpus{num_cpus}.csv"
    num_runs = 16
    header = "Boxcar Chunk Width,Normalize Chunk Width," + ",".join(f"Run {i+1}" for i in range(num_runs)) + ",Average Time"
    np.savetxt(filename, times, delimiter=",", header=header, fmt='%s', comments='')
    print(f"Data saved to {filename}")

def plot_heatmap(times):
    """Plot and save a heatmap of the times data."""
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(times[:, -1 - num_runs].reshape(int(np.sqrt(len(times))), int(np.sqrt(len(times)))), cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(heatmap, label='Execution Time (seconds)')
    plt.xlabel('Boxcar Chunk Width')
    plt.ylabel('Normalize Chunk Width')
    plt.title('Execution Time Heatmap')
    plt.savefig('heatmap.png')
    plt.close()
    print("Heatmap saved to 'heatmap.png'.")

# Example usage
if __name__ == "__main__":
    min_width = 1024
    max_width = 65536
    step = 1024
    num_cpus = 8
    times = find_optimal_combinations(min_width, max_width, step, num_cpus)
    save_times_data(times, num_cpus)
    plot_heatmap(times)
