#include "mdct_cuda.hpp"

#include <cuda_runtime.h>
#include <iostream>

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_LAST_CUDA_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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
  int blockSize = 512; // Number of threads per block
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


__global__ void postRotationKernel(var_t *d_out, 
                                 const var_t *t, 
                                 int N2, int N4, 
                                 int shift,
                                 var_t sine, 
                                 int overlap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (N4 + 1) >> 1) {
        var_t re, im, yr, yi;
        var_t t0, t1;
        
        // Calculate left pointer position
        var_t *yp0 = d_out + (overlap >> 1) + 2 * i;
        // Calculate right pointer position
        var_t *yp1 = d_out + (overlap >> 1) + N2 - 2 - 2 * i;
        
        // Process the first pair of values
        re = yp0[0];
        im = yp0[1];
        t0 = t[i << shift];
        t1 = t[(N4 - i) << shift];
        yr = S_MUL(re, t0) - S_MUL(im, t1);
        yi = S_MUL(im, t0) + S_MUL(re, t1);
        
        // Save the first pair of results
        var_t yr1 = yr;
        var_t yi1 = yi;
        
        // Process the second pair of values
        re = yp1[0];
        im = yp1[1];
        t0 = t[(N4 - i - 1) << shift];
        t1 = t[(i + 1) << shift];
        yr = S_MUL(re, t0) - S_MUL(im, t1);
        yi = S_MUL(im, t0) + S_MUL(re, t1);
        
        // Write results in the same order as the CPU version
        yp0[0] = -(yr1 - S_MUL(yi1, sine));  // Left real
        yp1[1] = yi1 + S_MUL(yr1, sine);     // Right imag
        yp1[0] = -(yr - S_MUL(yi, sine));    // Right real
        yp0[1] = yi + S_MUL(yr, sine);       // Left imag
    }
}

__global__ void mirrorKernel(var_t *d_out, 
                           const var_t *window,
                           int overlap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < overlap / 2) {
        var_t x1, x2;
        var_t *xp1 = d_out + overlap - 1 - i;
        var_t *yp1 = d_out + i;
        const var_t *wp1 = window + i;
        const var_t *wp2 = window + overlap - 1 - i;
        
        x1 = *xp1;
        x2 = *yp1;
        
        // Use temporary variables to avoid writing order issues
        var_t temp1 = S_MUL(*wp2, x2) - S_MUL(*wp1, x1);
        var_t temp2 = S_MUL(*wp1, x2) + S_MUL(*wp2, x1);
        
        *yp1 = temp1;
        *xp1 = temp2;
    }
}


void postAndMirrorWithCuda(var_t *out, const var_t *t, int N2, int N4, int shift, 
                          int stride, var_t sine, int overlap, const var_t *window) {
    var_t *d_out, *d_t, *d_window;
    
    // Allocate memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_out, (N2 + overlap) * sizeof(var_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_t, (N4 << shift) * sizeof(var_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_window, overlap * sizeof(var_t)));
    
    // Copy input data
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, out, (N2 + overlap) * sizeof(var_t), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_t, t, (N4 << shift) * sizeof(var_t), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_window, window, overlap * sizeof(var_t), 
                               cudaMemcpyHostToDevice));
    
    const int blockSize = 256;
    
    // post-rotation kernel
    int numElementsRotation = (N4 + 1) >> 1;
    int numBlocksRotation = (numElementsRotation + blockSize - 1) / blockSize;
    postRotationKernel<<<numBlocksRotation, blockSize>>>(d_out, d_t, N2, N4, 
                                                      shift, sine, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    
    // mirror kernel
    int numElementsMirror = overlap / 2;
    int numBlocksMirror = (numElementsMirror + blockSize - 1) / blockSize;
    mirrorKernel<<<numBlocksMirror, blockSize>>>(d_out, d_window, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    
    // Copy results
    CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, (N2 + overlap) * sizeof(var_t), 
                               cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_out);
    cudaFree(d_t);
    cudaFree(d_window);
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



