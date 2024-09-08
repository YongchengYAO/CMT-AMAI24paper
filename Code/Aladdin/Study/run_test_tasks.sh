#!/bin/bash

# Directory containing the scripts
script_dir="$(pwd)/test-tasks"

# Loop through all .sh files in the directory
for script in "$script_dir"/*.sh; do
    # Check if the file exists and is a regular file
    if [ -f "$script" ]; then
	echo "----------------------------------------"
        # Print the script name
        echo "Running script: $(basename "$script")"
        
        # Execute the script
        bash "$script"
        
        # Print a separator for better readability
        echo "----------------------------------------"
    fi
done

echo "All scripts have been executed."
