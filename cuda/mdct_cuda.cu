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
                                   const var_t *t, int N2, int N4, int shift,
                                   var_t sine, int overlap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (N4 + 1) >> 1) {
        var_t re, im, yr, yi;
        var_t t0, t1;

        // Accessing the output pointers
        var_t *yp0 = d_out + (overlap >> 1) + 2 * i;
        var_t *yp1 = d_out + (overlap >> 1) + (N2 - 2) - 2 * i;

        re = yp0[0];
        im = yp0[1];
        t0 = t[i << shift];
        t1 = t[(N4 - i) << shift];

        yr = S_MUL(re, t0) - S_MUL(im, t1);
        yi = S_MUL(im, t0) + S_MUL(re, t1);

        yp0[0] = -(yr - S_MUL(yi, sine));
        yp0[1] = yi + S_MUL(yr, sine);

        re = yp1[0];
        im = yp1[1];

        t0 = t[(N4 - i - 1) << shift];
        t1 = t[(i + 1) << shift];

        yr = S_MUL(re, t0) - S_MUL(im, t1);
        yi = S_MUL(im, t0) + S_MUL(re, t1);

        yp1[0] = -(yr - S_MUL(yi, sine));
        yp1[1] = yi + S_MUL(yr, sine);
    }
}

__global__ void mirrorKernel(var_t *d_out, const var_t *window,
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

        *yp1 = S_MUL(*wp2, x2) - S_MUL(*wp1, x1);
        *xp1 = S_MUL(*wp1, x2) + S_MUL(*wp2, x1);
    }
}

void postAndMirrorWithCuda(var_t *out, const var_t *t, int N2, int N4, int shift, int stride, var_t sine, int overlap, const var_t *window) {


    var_t *d_out, *d_t, *d_window;
    cudaMalloc((void **)&d_out, (N2 + overlap) * sizeof(var_t));
    cudaMalloc((void **)&d_t, (N4 << shift) * sizeof(var_t));
    cudaMalloc((void **)&d_window, overlap * sizeof(var_t));

    cudaMemcpy(d_out, out, (N2 + overlap) * sizeof(var_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, t, (N4 << shift) * sizeof(var_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_window, window, overlap * sizeof(var_t), cudaMemcpyHostToDevice);

    int blockSize = 512; // Number of threads per block
    int numBlocks = 1; // just use one block

    // Launch post-rotation kernel
    postRotationKernel<<<numBlocks, blockSize>>>(d_out, d_t, N2, N4, shift, sine, overlap);
    cudaDeviceSynchronize();

    // Launch mirror kernel
    mirrorKernel<<<numBlocks, blockSize>>>(d_out, d_window, overlap);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, (N2 + overlap) * sizeof(var_t), cudaMemcpyDeviceToHost);
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



