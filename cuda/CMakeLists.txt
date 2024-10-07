cmake_minimum_required(VERSION 3.16)

project(libnyquist_cuda LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80)

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
add_executable(libnyquist_cuda_test test.cpp)
target_link_libraries(libnyquist_cuda_test libnyquist_cuda)
