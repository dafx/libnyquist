#!/bin/sh

# Function to profile a specific kernel
profile_kernel() {
    local kernel_name=$1
    local output_file=$2
    
    echo "Profiling kernel: $kernel_name"
    ncu --metrics gpu__time_duration.avg,sm__throughput.avg,dram__throughput.avg,\
l1tex__t_bytes.avg,sm__warps_active.avg,\
sm__pipe_alu_cycles_active.avg,sm__pipe_fma_cycles_active.avg \
        --kernel-name "$kernel_name" \
        --csv \
        --target-processes all \
        -c 10 \
        ./out/build/gcc-cuda/bin/libnyquist-examples test_data/sb-reverie.opus > "$output_file" 2>&1
}

# Function to build and profile all kernels for a branch
build_and_profile_branch() {
    local branch=$1
    local output_dir=$2
    local kernel1=$3
    local kernel2=$4
    local kernel3=$5
    
    echo "Checking out branch: $branch"
    git checkout $branch
    
    echo "Building..."
    cmake --workflow --preset gcc-cuda
    
    echo "Running NCU profiling for all kernels..."
    mkdir -p "$output_dir"
    
    profile_kernel "$kernel1" "$output_dir/kernel1_profile.csv"
    profile_kernel "$kernel2" "$output_dir/kernel2_profile.csv"
    profile_kernel "$kernel3" "$output_dir/kernel3_profile.csv"
}

# Create base directory for reports
mkdir -p benchmark/kernel_comparison

# Profile unfused version (multichannel branch)
build_and_profile_branch "multichannel" \
    "benchmark/kernel_comparison/unfused" \
    "doPreRotation" \
    "regular_fft" \
    "postAndMirrorKernel"

# Profile fused version
build_and_profile_branch "6-fuse-2c-pre-and-2c-postmirror" \
    "benchmark/kernel_comparison/fused" \
    "doPreRotationFused" \
    "regular_fft" \
    "postAndMirrorKernelFused"

# Print completion message
echo "Kernel profiling completed for both versions."
echo "Results are saved in:"
echo "  - Unfused kernels: benchmark/kernel_comparison/unfused/"
echo "  - Fused kernels: benchmark/kernel_comparison/fused/"
