#include "mdct_cuda.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

// Error checking macro
#define CHECK_CUDA(call) do {				\
    cudaError_t err = call;				\
    if (err != cudaSuccess) {				\
      fprintf(stderr, "CUDA error %d: %s at %s:%d\n",   \
	      err, cudaGetErrorString(err),             \
	      __FILE__, __LINE__);                      \
      return;                                           \
    }							\
  } while(0)


// Pre-rotation kernel
__global__ void preRotationKernel(const var_t *in, var_t *f2,
				  const var_t *trig, int N, int N2, int N4,
				  int shift, int stride, var_t sine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N4) {
    int in_idx1 = idx * 2 * stride;
    int in_idx2 = stride * (N2 - 1) - idx * 2 * stride;

    if (in_idx1 >= 0 && in_idx1 < N * stride &&
	in_idx2 >= 0 && in_idx2 < N * stride) {
      const var_t x1 = in[in_idx1];
      const var_t x2 = in[in_idx2];

      int trig_idx1 = idx << shift;
      int trig_idx2 = (N4 - idx) << shift;

      if (trig_idx1 < N && trig_idx2 < N) {
	var_t yr = -S_MUL(x2, trig[trig_idx1]) +
	  S_MUL(x1, trig[trig_idx2]);
	var_t yi = -S_MUL(x2, trig[trig_idx2]) -
	  S_MUL(x1, trig[trig_idx1]);

	int f2_idx = idx * 2;
	if (f2_idx + 1 < N2) {
	  f2[f2_idx] = yr - S_MUL(yi, sine);
	  f2[f2_idx + 1] = yi + S_MUL(yr, sine);
	}
      }
    }
  }
}


// Post-rotation and de-shuffle kernel
__global__ void postRotationKernel(var_t *out, var_t *f2,
				   const var_t *trig, int N, int N2, int N4,
				   int overlap, int shift, var_t sine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < (N4 + 1) >> 1) {
    const int base_offset = overlap >> 1;
    int yp0_idx = base_offset + idx * 2;
    int yp1_idx = base_offset + N2 - 2 - idx * 2;

    if (yp0_idx + 1 < N && yp1_idx + 1 < N &&
	yp0_idx >= 0 && yp1_idx >= 0) {
      var_t re, im, yr, yi;
      var_t t0, t1;

      // First pair
      re = f2[idx * 2];
      im = f2[idx * 2 + 1];
      t0 = trig[idx << shift];
      t1 = trig[(N4 - idx) << shift];

      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      out[yp0_idx] = -(yr - S_MUL(yi, sine));
      out[yp1_idx + 1] = yi + S_MUL(yr, sine);

      // Second pair
      re = f2[N2 - 2 - idx * 2];
      im = f2[N2 - 2 - idx * 2 + 1];
      t0 = trig[(N4 - idx - 1) << shift];
      t1 = trig[(idx + 1) << shift];

      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      out[yp1_idx] = -(yr - S_MUL(yi, sine));
      out[yp0_idx + 1] = yi + S_MUL(yr, sine);
    }
  }
}


// Mirror for TDAC kernel
__global__ void mirrorKernel(var_t *out, const var_t *window,
			     int N, int overlap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < overlap / 2) {
    int out_idx1 = idx;
    int out_idx2 = overlap - 1 - idx;

    if (out_idx1 < N && out_idx2 < N &&
	out_idx1 >= 0 && out_idx2 >= 0 && idx < overlap) {
      var_t x1 = out[out_idx2];
      var_t x2 = out[out_idx1];
      var_t w1 = window[idx];
      var_t w2 = window[overlap - 1 - idx];

      out[out_idx1] = S_MUL(w2, x2) - S_MUL(w1, x1);
      out[out_idx2] = S_MUL(w1, x2) + S_MUL(w2, x1);
    }
  }
}

// Main CUDA implementation function
void mdctBackwardWithCuda(const var_t *in, var_t *out, const var_t *trig,
			  const var_t *window, int N, int overlap, int shift,
			  int stride) {
  // Input validation
  if (!in || !out || !trig || !window) {
    fprintf(stderr, "Invalid input pointers\n");
    return;
  }

  if (N <= 0 || overlap <= 0 || shift < 0 || stride <= 0) {
    fprintf(stderr, "Invalid parameters\n");
    return;
  }

  // Adjust N based on shift
  N >>= shift;
  int N2 = N >> 1;
  int N4 = N >> 2;

  // Calculate sine value
  var_t sine = (var_t)(2 * 3.14159265358979323846 * 0.125) / N;

  // Allocate device memory
  var_t *d_in, *d_out, *d_trig, *d_window, *d_f2;
  size_t in_size = N2 * stride * sizeof(var_t);
  size_t out_size = N2 * sizeof(var_t);
  size_t trig_size = (N4 << shift) * sizeof(var_t);
  size_t window_size = overlap * sizeof(var_t);
  size_t f2_size = N2 * sizeof(var_t);

  // Allocate and copy memory to device
  CHECK_CUDA(cudaMalloc(&d_in, in_size));
  CHECK_CUDA(cudaMalloc(&d_out, out_size));
  CHECK_CUDA(cudaMalloc(&d_trig, trig_size));
  CHECK_CUDA(cudaMalloc(&d_window, window_size));
  CHECK_CUDA(cudaMalloc(&d_f2, f2_size));

  CHECK_CUDA(cudaMemcpy(d_in, in, in_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_trig, trig, trig_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_window, window, window_size, cudaMemcpyHostToDevice));

  // Create CUFFT plan and execute
  cufftHandle plan;
  cufftPlan1d(&plan, N4, CUFFT_C2C, 1);

  // Launch kernels
  int blockSize = 512;
  int numBlocks;

  // 1. Pre-rotation
  numBlocks = (N4 + blockSize - 1) / blockSize;
  preRotationKernel<<<numBlocks, blockSize>>>(d_in, d_f2, d_trig,
					      N, N2, N4, shift,
					      stride, sine);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 2. IFFT using cuFFT
  cufftExecC2C(plan, (cufftComplex*)d_f2,
	       (cufftComplex*)(d_out + (overlap >> 1)),
	       CUFFT_INVERSE);

  CHECK_CUDA(cudaDeviceSynchronize());

  // 3. Post-rotation
  numBlocks = ((N4 + 1) >> 1 + blockSize - 1) / blockSize;
  postRotationKernel<<<numBlocks, blockSize>>>(d_out, d_f2, d_trig,
					       N, N2, N4, overlap,
					       shift, sine);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 4. Mirror
  numBlocks = (overlap/2 + blockSize - 1) / blockSize;
  mirrorKernel<<<numBlocks, blockSize>>>(d_out, d_window,
					 N, overlap);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost));

  // Cleanup
  cufftDestroy(plan);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_trig);
  cudaFree(d_window);
  cudaFree(d_f2);
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
