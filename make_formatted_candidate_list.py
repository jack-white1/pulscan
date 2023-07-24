import numpy as np
from scipy.stats import norm
import argparse
import os
from scipy.stats import chi2 as chi2_distribution
import numpy as np

def extended_equiv_gaussian_sigma(logp):
    t = np.sqrt(-2.0 * logp)
    num = 2.515517 + t * (0.802853 + t * 0.010328)
    denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))
    return t - num / denom

def log_asymptotic_incomplete_gamma(a, z):
    x = 1.0
    newxpart = 1.0
    term = 1.0
    ii = 1
    while np.abs(newxpart) > 1e-15:
        term *= (a - ii)
        newxpart = term / np.power(z, ii)
        x += newxpart
        ii += 1
    return (a - 1.0) * np.log(z) - z + np.log(x)

def log_asymptotic_gamma(z):
    x = (z - 0.5) * np.log(z) - z + 0.91893853320467267
    y = 1.0 / (z * z)
    x += (((-5.9523809523809529e-4 * y
            + 7.9365079365079365079365e-4) * y
           - 2.7777777777777777777778e-3) * y + 8.3333333333333333333333e-2) / z
    return x

def equivalent_gaussian_sigma(logp):
    x = 0.0
    if logp < -600.0:
        x = extended_equiv_gaussian_sigma(logp)
    else:
        q = np.exp(logp)
        p = 1.0 - q
        x = norm.ppf(p)
        if x < 0.0:
            x = 0.0
    return x

def chi2_logp(chi2, dof):
    logp = -np.inf
    if chi2 > 0.0:
        if chi2 / dof > 15.0 or (dof > 150 and chi2 / dof > 6.0):
            logp = log_asymptotic_incomplete_gamma(0.5 * dof, 0.5 * chi2) - log_asymptotic_gamma(0.5 * dof)
        else:
            p = chi2_distribution.cdf(chi2, dof)  # Corrected here
            logp = np.log(1.0 - p)
    return logp


def chi2_sigma(chi2, dof):
    if chi2 <= 0.0:
        return 0.0
    logp = chi2_logp(chi2, dof)
    return equivalent_gaussian_sigma(logp)

def presto_candidate_sigma(power, numsum, numtrials):
    if power <= 0.0:
        return 0.0
    chi2 = 2.0 * power
    dof = 2.0 * numsum
    logp = chi2_logp(chi2, dof)
    logp += np.log(numtrials)
    return equivalent_gaussian_sigma(logp)

def fdot_from_boxcar_width(boxcar_width):
    return boxcar_width / (observation_time**2)

def acceleration_from_fdot(fdot, frequency):
    return fdot * speed_of_light / frequency

# Create an argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('input_file', metavar='input_file', type=str, help='the input file to process')

# Parse the command line arguments
args = parser.parse_args()

# Load data from the input file specified as a command line argument
data = np.loadtxt(args.input_file, delimiter=',', skiprows=1)

numtrials = 1200 / 6.95
sample_time = 128e-6
num_samples = 2352000*2
observation_time = sample_time * num_samples
speed_of_light = 299792458

sampling_frequency = 1 / sample_time
frequency_bin_width = sampling_frequency / num_samples

# Prepare an array to store results.
results = []

# for each row in the data
for row in data:
    frequency_index = row[1]
    frequency = frequency_index * frequency_bin_width
    period_ms = 1000 / frequency

    boxcar_width = row[0]
    fdot = fdot_from_boxcar_width(boxcar_width)
    acceleration = acceleration_from_fdot(fdot, frequency)

    chi2_value = row[2]

    k = boxcar_width * 2    # degrees of freedom
    p_value_scipy = chi2_distribution.sf(chi2_value, k)
    trials_corrected_p_value_scipy = p_value_scipy * numtrials # bonferroni correction
    if trials_corrected_p_value_scipy == 0:
        p_value_scipy = chi2_logp(chi2_value, k)
        trials_corrected_z_score = presto_candidate_sigma(chi2_value, boxcar_width, numtrials)
    else:
        trials_corrected_z_score = -1 * norm.ppf(trials_corrected_p_value_scipy)

    # Append the result to results array
    results.append([trials_corrected_z_score, chi2_value, period_ms, frequency, frequency_index, fdot, boxcar_width, acceleration])

# Convert results to a numpy array
results_np = np.array(results)

# Get directory and file name without extension from input file
directory, filename = os.path.split(args.input_file)
name, extension = os.path.splitext(filename)

# Define output file paths
csv_output_path = os.path.join(directory, f"{name}_unformatted.csv")
txt_output_path = os.path.join(directory, f"{name}_formatted.txt")

# Write the results to a file in csv format of boxcar_width, frequency_index, trials_corrected_z_score
np.savetxt(csv_output_path, results_np, delimiter=',', header="trials_corrected_z_score,chi2_value,period_ms,frequency,frequency_index,fdot,boxcar_width,acceleration", comments="")

# Sort the data by trials_corrected_z_score (column index 0)
sort_indices = np.argsort(-results_np[:, 0])
results_np = results_np[sort_indices]

# Prepare an array to store the reformatted results
results = []

# Define the header lines
header1 = "               Summed   Coherent  Num        Period          Frequency         FFT 'r'        Freq Deriv       FFT 'z'         Accel"
header2 = "Cand    Sigma  Power    Power     Harm       (ms)            (Hz)              (bin)          (Hz/s)           (bins)          (±m/s^2)"
header3 = "---------------------------------------------------------------------------------------------------------------------------------------"

# Add headers to the results array
results.extend([header1, header2, header3])

numharm = 1

# Reformat each row of data and add it to the results array
for i, row in enumerate(results_np, 1):
    trials_corrected_z_score, chi2_value, period_ms, frequency, frequency_index, fdot, boxcar_width, acceleration = row
    line = "{:<8}{:<7.2f}{:<9.2f}{:<10.2f}{:<11}{:<16.7f}{:<18.7f}{:<15.2f}{:<17.7f}{:<16.2f}{:<14.2f}".format(i, trials_corrected_z_score, chi2_value, chi2_value, numharm, period_ms, frequency, frequency_index, fdot, boxcar_width, acceleration)
    results.append(line)

# Write the results to a new text file
with open(txt_output_path, 'w') as f:
    f.write("\n".join(results))