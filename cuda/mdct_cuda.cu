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

// Combined CUDA kernel for the entire MDCT backward transform
__global__ void mdctBackwardKernel(const var_t *in, var_t *out,
				   const var_t *trig, const var_t *window,
				   var_t *f2, int N, int N2, int N4,
				   int overlap, int shift, int stride,
				   var_t sine) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Pre-rotate
  if (idx < N4) {
    int in_idx1 = idx * 2 * stride;
    int in_idx2 = stride * (N2 - 1) - idx * 2 * stride;

    if (in_idx1 >= 0 && in_idx1 < N * stride && in_idx2 >= 0 &&
	in_idx2 < N * stride) {
      const var_t x1 = in[in_idx1];
      const var_t x2 = in[in_idx2];

      int trig_idx1 = idx << shift;
      int trig_idx2 = (N4 - idx) << shift;

      if (trig_idx1 < N && trig_idx2 < N) {
	var_t yr = -S_MUL(x2, trig[trig_idx1]) + S_MUL(x1, trig[trig_idx2]);
	var_t yi = -S_MUL(x2, trig[trig_idx2]) - S_MUL(x1, trig[trig_idx1]);

	int f2_idx = idx * 2;
	if (f2_idx + 1 < N2) {
	  f2[f2_idx] = yr - S_MUL(yi, sine);
	  f2[f2_idx + 1] = yi + S_MUL(yr, sine);
	}
      }
    }
  }
  __syncthreads();

  // Post-rotate and de-shuffle
  if (idx < (N4 + 1) >> 1) {
    const int base_offset = overlap >> 1;
    int yp0_idx = base_offset + idx * 2;
    int yp1_idx = base_offset + N2 - 2 - idx * 2;

    if (yp0_idx + 1 < N && yp1_idx + 1 < N && yp0_idx >= 0 && yp1_idx >= 0) {

      int f2_idx1 = idx * 2;
      int f2_idx2 = N2 - 2 - idx * 2;

      if (f2_idx1 + 1 < N2 && f2_idx2 + 1 < N2) {
	var_t re = f2[f2_idx1];
	var_t im = f2[f2_idx1 + 1];

	int trig_idx1 = idx << shift;
	int trig_idx2 = (N4 - idx) << shift;

	if (trig_idx1 < N && trig_idx2 < N) {
	  var_t t0 = trig[trig_idx1];
	  var_t t1 = trig[trig_idx2];

	  var_t yr = S_MUL(re, t0) - S_MUL(im, t1);
	  var_t yi = S_MUL(im, t0) + S_MUL(re, t1);

	  out[yp0_idx] = -(yr - S_MUL(yi, sine));
	  out[yp1_idx + 1] = yi + S_MUL(yr, sine);

	  // Second pair
	  re = f2[f2_idx2];
	  im = f2[f2_idx2 + 1];

	  trig_idx1 = (N4 - idx - 1) << shift;
	  trig_idx2 = (idx + 1) << shift;

	  if (trig_idx1 < N && trig_idx2 < N) {
	    t0 = trig[trig_idx1];
	    t1 = trig[trig_idx2];

	    yr = S_MUL(re, t0) - S_MUL(im, t1);
	    yi = S_MUL(im, t0) + S_MUL(re, t1);

	    out[yp1_idx] = -(yr - S_MUL(yi, sine));
	    out[yp0_idx + 1] = yi + S_MUL(yr, sine);
	  }
	}
      }
    }
  }
  __syncthreads();

  // Mirror for TDAC
  if (idx < overlap / 2) {
    int out_idx1 = idx;
    int out_idx2 = overlap - 1 - idx;

    if (out_idx1 < N && out_idx2 < N && out_idx1 >= 0 && out_idx2 >= 0 &&
	idx < overlap) {
      var_t x1 = out[out_idx2];
      var_t x2 = out[out_idx1];
      var_t w1 = window[idx];
      var_t w2 = window[overlap - 1 - idx];

      out[out_idx1] = S_MUL(w2, x2) - S_MUL(w1, x1);
      out[out_idx2] = S_MUL(w1, x2) + S_MUL(w2, x1);
    }
  }
}

void mdctBackwardWithCuda(const var_t *in, var_t *out, const var_t *trig,
			  const var_t *window, int N, int overlap, int shift,
			  int stride) {
  if (!in || !out || !trig || !window) {
    std::cerr << "Invalid input pointers" << std::endl;
    return;
  }

  if (N <= 0 || overlap <= 0 || shift < 0 || stride <= 0) {
    std::cerr << "Invalid parameters" << std::endl;
    return;
  }

  int N2 = N >> 1;
  int N4 = N >> 2;
  var_t sine = (var_t)(2 * PI * 0.125) / N;

  int blockSize = 256;
  int numBlocks = (N4 + blockSize - 1) / blockSize;

  size_t in_size = N2 * stride * sizeof(var_t);
  size_t out_size = N * sizeof(var_t);
  size_t trig_size = (N4 << shift) * sizeof(var_t);
  size_t window_size = overlap * sizeof(var_t);
  size_t f2_size = N2 * sizeof(var_t);

  // memories in device
  var_t *d_in = nullptr;  // input, should copy from host
  var_t *d_out = nullptr; // output, should copy to host
  var_t *d_trig = nullptr; // trig table, should copy from host
  var_t *d_window = nullptr; // window, should copy from host
  var_t *d_f2 = nullptr; // temporary in device

  cudaError_t err;

#if 0
  std::cout << "Allocating memory..." << std::endl;
  std::cout << "in_size: " << in_size << std::endl;
  std::cout << "out_size: " << out_size << std::endl;
  std::cout << "trig_size: " << trig_size << std::endl;
  std::cout << "window_size: " << window_size << std::endl;
  std::cout << "f2_size: " << f2_size << std::endl;
#endif

  if ((err = cudaMalloc(&d_in, in_size)) != cudaSuccess ||
      (err = cudaMalloc(&d_out, out_size)) != cudaSuccess ||
      (err = cudaMalloc(&d_trig, trig_size)) != cudaSuccess ||
      (err = cudaMalloc(&d_window, window_size)) != cudaSuccess ||
      (err = cudaMalloc(&d_f2, f2_size)) != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    goto cleanup;
  }

  if ((err = cudaMemset(d_out, 0, out_size)) != cudaSuccess ||
      (err = cudaMemset(d_f2, 0, f2_size)) != cudaSuccess) {
    std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
    goto cleanup;
  }

  if ((err = cudaMemcpy(d_in, in, in_size, cudaMemcpyHostToDevice)) !=
	  cudaSuccess ||
      (err = cudaMemcpy(d_trig, trig, trig_size, cudaMemcpyHostToDevice)) !=
	  cudaSuccess ||
      (err = cudaMemcpy(d_window, window, window_size,
			cudaMemcpyHostToDevice)) != cudaSuccess) {
    std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err)
	      << std::endl;
    goto cleanup;
  }

#ifdef DEBUG
  std::cout << "Launching kernel with blocks=" << numBlocks
	    << ", threads=" << blockSize << std::endl;
#endif

  mdctBackwardKernel<<<numBlocks, blockSize>>>(d_in, d_out, d_trig, d_window,
					       d_f2, N, N2, N4, overlap, shift,
					       stride, sine);

  if ((err = cudaGetLastError()) != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
	      << std::endl;
    goto cleanup;
  }

  if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
	      << std::endl;
    goto cleanup;
  }

  if ((err = cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost)) !=
      cudaSuccess) {
    std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err)
	      << std::endl;
  }

cleanup:
  if (d_in)
    cudaFree(d_in);
  if (d_out)
    cudaFree(d_out);
  if (d_trig)
    cudaFree(d_trig);
  if (d_window)
    cudaFree(d_window);
  if (d_f2)
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
