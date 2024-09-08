#!/bin/bash

# Specify the directory containing the MATLAB scripts
# Update this path to the directory containing your MATLAB `.m` files
SCRIPT_DIR="eval-surfAreaRelDiff"

# Loop through all `.m` files in the directory
for script_file in "${SCRIPT_DIR}"/*.m; do
    echo "Running MATLAB script $script_file"
    # Run MATLAB script
    # Ensure MATLAB binary is in your PATH or specify the full path to Matlab binary
    matlab -batch "run('${script_file}'); exit;"
done

