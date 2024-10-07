#pragma once

#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif

CUDA_KERNEL void doPreRotation(float *input, float *output, int N);

void printCudaVersion();
