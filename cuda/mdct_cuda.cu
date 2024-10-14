#include "mdct_cuda.hpp"

#include <cuda_runtime.h>
#include <iostream>

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

// CUDA kernel
__global__ void doPreRotation(const var_t *xp1, var_t *yp, const var_t *t,
                              int N4, int shift, int stride, int N2,
                              var_t sine) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the thread index

  if (i < N4) {
    const var_t *xp1_i = xp1 + i * 2 * stride;
    const var_t *xp2_i = xp1 + stride * (N2 - 1) - i * 2 * stride;

    // Calculate yr and yi for each thread
    var_t yr, yi;
    yr = -S_MUL(*xp2_i, t[i << shift]) + S_MUL(*xp1_i, t[(N4 - i) << shift]);
    yi = -S_MUL(*xp2_i, t[(N4 - i) << shift]) - S_MUL(*xp1_i, t[i << shift]);

    // Store results
    yp[i * 2] = yr - S_MUL(yi, sine);
    yp[i * 2 + 1] = yi + S_MUL(yr, sine);
  }
}

// Host code with memory management
void preRotateWithCuda(const var_t *host_xp1, var_t *host_yp,
                       const var_t *host_t, int N, int shift, int stride,
                       var_t sine) {

  int N2 = N >> 1;
  int N4 = N >> 2;
  // Device pointers
  var_t *dev_xp1;
  var_t *dev_yp;
  var_t *dev_t;

  // Allocate memory on the device (GPU)
  cudaMalloc((void **)&dev_xp1, N4 * 2 * stride * sizeof(var_t));
  cudaMalloc((void **)&dev_yp, N4 * 2 * sizeof(var_t));
  cudaMalloc((void **)&dev_t, (N4 << shift) * sizeof(var_t));

  // Copy input data from host (CPU) to device (GPU)
  cudaMemcpy(dev_xp1, host_xp1, N4 * 2 * stride * sizeof(var_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_t, host_t, (N4 << shift) * sizeof(var_t),
             cudaMemcpyHostToDevice);

  // Launch the kernel with the appropriate block and grid sizes
  int blockSize = 32; // Number of threads per block
  int numBlocks = (N4 + blockSize - 1) /
                  blockSize; // Number of blocks, ensuring full coverage

  doPreRotation<<<numBlocks, blockSize>>>(dev_xp1, dev_yp, dev_t, N4, shift,
                                          stride, N2, sine);

  // Synchronize to ensure kernel execution is complete
  cudaDeviceSynchronize();

  // Copy the result back to the host (CPU)
  cudaMemcpy(host_yp, dev_yp, N4 * 2 * sizeof(var_t), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(dev_xp1);
  cudaFree(dev_yp);
  cudaFree(dev_t);
}

void printCudaVersion() {
  std::cout << "CUDA Compiled version: " << __CUDACC_VER_MAJOR__ << std::endl;

  int runtime_ver;
  cudaRuntimeGetVersion(&runtime_ver);
  std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

  int driver_ver;
  cudaDriverGetVersion(&driver_ver);
  std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}
