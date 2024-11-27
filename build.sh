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
    # Profile all three kernels with metrics
    ncu --metrics gpu__time_duration.avg,sm__throughput.avg,dram__throughput.avg,\
l1tex__t_bytes.avg,sm__warps_active.avg,\
sm__pipe_alu_cycles_active.avg,sm__pipe_fma_cycles_active.avg \
        --kernel-name "doPreRotation","regular_fft","postAndMirrorKernel","doPreRotationFused","postAndMirrorKernelFused" \
        --csv \
        --target-processes all \
        -c 10 \
        ./out/build/gcc-cuda/bin/libnyquist-examples test_data/sb-reverie.opus > "$output_file" 2>&1
}

# Create directory for reports if it doesn't exist
mkdir -p benchmark/kernel_comparison

# Profile unfused version (multichannel branch)
build_and_profile "multichannel" "benchmark/kernel_comparison/unfused_kernels.csv"

# Profile fused version
build_and_profile "6-fuse-2c-pre-and-2c-postmirror" "benchmark/kernel_comparison/fused_kernels.csv"

# Print completion message
echo "Kernel profiling completed for both versions."
echo "Results are saved in:"
echo "  - Unfused kernels: benchmark/kernel_comparison/unfused_kernels.csv"
echo "  - Fused kernels: benchmark/kernel_comparison/fused_kernels.csv"
