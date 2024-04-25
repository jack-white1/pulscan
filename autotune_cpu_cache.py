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
    """Run the pulscan command with specified boxcar chunk widths and return the normalization and search times."""
    command = [
        "./pulscan", "sample_data/1298us_binary.fft",
        "-chunkwidth", str(boxcar_chunk_width),
        "-normalizechunkwidth", str(normalize_chunk_width),
        "-ncpus", str(num_cpus),
        "-zmax", str(1024)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    normalization_time_search = re.search(r"Normalizing the data took\s+([0-9.]+) seconds", output)
    search_time_search = re.search(r"Searching the data took\s+([0-9.]+) seconds", output)

    if normalization_time_search and search_time_search:
        normalization_time = float(normalization_time_search.group(1))
        search_time = float(search_time_search.group(1))
        return normalization_time, search_time
    else:
        raise ValueError("Time outputs not found in the response")

def find_optimal_widths(max_width, num_cpus):
    """Test each width for normalization and searching separately."""
    widths = [256 * (2 ** i) for i in range(int(math.log(max_width / 256, 2)) + 1)]
    num_runs = 16
    normalization_times = []
    search_times = []

    for width in widths:
        for i in range(num_runs):
            try:
                normalization_time, search_time = run_pulscan(width, width, num_cpus)
                normalization_times.append([width, i+1, normalization_time])
                search_times.append([width, i+1, search_time])
                print(f"Experiment {i+1}, Width = {width}, Normalization Time = {normalization_time}s, Search Time = {search_time}s")
            except ValueError as e:
                print(f"Error: {e}")
                normalization_times.append([width, i+1, None])
                search_times.append([width, i+1, None])

    return normalization_times, search_times

def save_times_data(normalization_times, search_times, num_cpus):
    """Save the times data to a CSV file named by the CPU model and num_cpus."""
    cpu_model = get_cpu_model()
    normalization_filename = f"normalization_times_{cpu_model}_ncpus{num_cpus}.csv"
    search_filename = f"search_times_{cpu_model}_ncpus{num_cpus}.csv"

    for times, filename, label in [(normalization_times, normalization_filename, "Normalization Time"), 
                                   (search_times, search_filename, "Search Time")]:
        header = ["Width", "Experiment Number", label]
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(times)
        print(f"Data saved to {filename}")

# Example usage
if __name__ == "__main__":
    max_width = 131072  # 128K
    num_cpus = 72
    normalization_times, search_times = find_optimal_widths(max_width, num_cpus)
    save_times_data(normalization_times, search_times, num_cpus)
