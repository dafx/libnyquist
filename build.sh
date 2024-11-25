#!/bin/sh

# Function to build and profile
build_and_profile() {
    local branch=$1
    local output_file=$2
    
    echo "Checking out branch: $branch"
    git checkout $branch
    
    echo "Building..."
    cmake --workflow --preset gcc-cuda
    
    echo "Running NCU profiling..."
    ncu --metrics gpu__time_duration.avg,sm__throughput.avg,dram__throughput.avg,\
l1tex__t_bytes.avg,sm__warps_active.avg,\
sm__pipe_alu_cycles_active.avg,sm__pipe_fma_cycles_active.avg \
        --kernel-regex "doPreRotation|doPreRotationFused" \
        --csv \
        --target-processes all \
        ./out/build/gcc-cuda/bin/libnyquist-examples test_data/sb-reverie.opus > "$output_file" 2>&1
}

# Create directory for reports if it doesn't exist
mkdir -p benchmark/report3/ncu_results

# Profile unfused version (multichannel branch)
build_and_profile "multichannel" "benchmark/report3/ncu_results/unfused_profile.txt"

# Profile fused version
build_and_profile "6-fuse-2c-pre-and-2c-postmirror" "benchmark/report3/ncu_results/fused_profile.txt"

# Compare and analyze results
echo "Analyzing results..."
echo "Performance Analysis Results" > benchmark/report3/ncu_results/analysis_report.txt
echo "=========================" >> benchmark/report3/ncu_results/analysis_report.txt
echo "" >> benchmark/report3/ncu_results/analysis_report.txt

# Extract and compare metrics
echo "Comparing metrics between unfused and fused versions:" >> benchmark/report3/ncu_results/analysis_report.txt
echo "1. Kernel Duration (gpu__time_duration.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "2. SM Throughput (sm__throughput.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "3. DRAM Throughput (dram__throughput.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "4. L1 Cache Transactions (l1tex__t_bytes.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "5. Warp Occupancy (sm__warps_active.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "6. ALU Utilization (sm__pipe_alu_cycles_active.avg)" >> benchmark/report3/ncu_results/analysis_report.txt
echo "7. FMA Utilization (sm__pipe_fma_cycles_active.avg)" >> benchmark/report3/ncu_results/analysis_report.txt

# Print completion message
echo "Performance analysis completed. Check benchmark/report3/ncu_results/ for detailed reports."
