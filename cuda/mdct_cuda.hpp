#pragma once

#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif

#ifdef __cplusplus
extern "C" {
#endif

CUDA_KERNEL void doPreRotation(float *input, float *output, int N);

#ifdef __cplusplus
}
#endif


void printCudaVersion();
