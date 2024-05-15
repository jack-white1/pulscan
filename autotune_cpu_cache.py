import subprocess
import re
import platform
import math
import time

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
    
    # Begin timer in python for overall command execution
    starttime = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stoptime = time.time()
    totaltime = stoptime - starttime
    output = result.stdout
    
    normalization_time_search = re.search(r"Normalizing the data took\s+([0-9.]+) seconds", output)
    search_time_search = re.search(r"Searching the data took\s+([0-9.]+) seconds", output)
    producing_output_time_search = re.search(r"Producing output took\s+([0-9.]+) seconds", output)
    
    if normalization_time_search and search_time_search and producing_output_time_search:
        normalization_time = float(normalization_time_search.group(1))
        search_time = float(search_time_search.group(1))
        producing_output_time = float(producing_output_time_search.group(1))
        return normalization_time, search_time, producing_output_time, totaltime
    else:
        raise ValueError("Time outputs not found in the response")

def find_optimal_widths(max_width, num_cpus):
    """Test each width for normalization and searching separately."""
    widths = [256 * (2 ** i) for i in range(int(math.log(max_width / 256, 2)) + 1)]
    num_runs = 8

    normalization_times = []
    search_times = []
    producing_output_times = []
    
    for width in widths:
        for i in range(num_runs):
            try:
                normalization_time, search_time, producing_output_time, totaltime = run_pulscan(width, width, num_cpus)
                normalization_times.append([width, i+1, normalization_time])
                search_times.append([width, i+1, search_time])
                producing_output_times.append([width, i+1, producing_output_time])
                
                #print(f"Experiment {i+1}, Width = {width}, Normalization Time = {normalization_time}s, Search Time = {search_time}s, Producing Output Time = {producing_output_time}s, Total Time = {totaltime}s")
            except ValueError as e:
                print(f"Error: {e}")
                normalization_times.append([width, i+1, None])
                search_times.append([width, i+1, None])
                producing_output_times.append([width, i+1, None])
        
        # Extract the search times and total times for averaging
        search_time_values = [entry[2] for entry in search_times if entry[2] is not None]
        total_time_values = [entry[2] for entry in producing_output_times if entry[2] is not None]

        if search_time_values and total_time_values:
            avg_search_time = sum(search_time_values) / len(search_time_values)
            avg_total_time = sum(total_time_values) / len(total_time_values)
            print(f"Width = {width}, average search time: {avg_search_time}, average total time: {avg_total_time}")

    return normalization_times, search_times, producing_output_times

# Example usage
if __name__ == "__main__":
    max_width = 131072  # 128K
    num_cpus = 16
    normalization_times, search_times, producing_output_times = find_optimal_widths(max_width, num_cpus)
