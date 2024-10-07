cmake_minimum_required(VERSION 3.16)

set(CUDAToolkit_ROOT /usr/local/cuda)

project(libnyquist_cuda LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the source files for the library
set(SOURCES
    mdct_cuda.cu
)

# Create the CUDA library
add_library(libnyquist_cuda STATIC ${SOURCES})

# Specify include directories
target_include_directories(libnyquist_cuda PUBLIC include)

# Set CUDA properties
set_target_properties(libnyquist_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link necessary libraries (if any)
# target_link_libraries(libnyquist_cuda some_other_library)