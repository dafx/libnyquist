#!/bin/bash  

# Constants
base='benchmark/memory-optimized'
optimized='constant_memory'
build_cmd="cmake --workflow --preset=gcc-cuda"
exe_path="./out/build/gcc-cuda/bin/libnyquist-examples"
test_file="./test_data/sb-reverie.opus"

# Function to run benchmark for a specific branch
run_benchmark() {  # Removed 'function' keyword - it's not needed in bash
    local branch="$1"  # Added quotes for better variable handling
    local output_file="$2"

    echo "Testing branch: $branch"
    git checkout "$branch"
    
    echo "Building the code..."
    $build_cmd
    
    echo "Running the benchmark..."
    "$exe_path" "$test_file" > "$output_file"
}

# Function to calculate average execution time from log file
calculate_average() {  # Removed 'function' keyword
    local log_file="$1"
    local avg
    avg=$(grep "Total Time:" "$log_file" | awk '{sum += $3; count++} END {if (count > 0) print sum/count}')
    echo "$avg"
}

# Ensure the benchmark directory exists
mkdir -p benchmark

# Run benchmarks for both branches
run_benchmark "$base" "benchmark/constant_base.log"
run_benchmark "$optimized" "benchmark/constant_optimized.log"

# Calculate averages
base_avg=$(calculate_average "benchmark/constant_base.log")
optimized_avg=$(calculate_average "benchmark/constant_optimized.log")

# Calculate and display results
echo "Base branch average time: $base_avg ms"
echo "Optimized branch average time: $optimized_avg ms"

# Calculate speedup
speedup=$(awk "BEGIN {print $base_avg/$optimized_avg}")
echo "Speedup: ${speedup}x"