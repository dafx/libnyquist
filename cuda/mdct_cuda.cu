#include "mdct_cuda.hpp"

__global__ void doPreRotationCuda(const float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float angle = M_PI / (2.0f * N) * (idx + 0.5f);
        output[idx] = input[idx] * cosf(angle) - input[N - 1 - idx] * sinf(angle); // Place holder dummy code, DO NOT RUN
    }
}

void doPreRotation(const float *input, float *output, int N)
{
}

#include <iostream>
void printCudaVersion()
{
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}
