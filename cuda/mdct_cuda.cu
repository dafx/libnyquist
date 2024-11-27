#include "mdct_cuda.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <float.h>
#include <iostream>

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_LAST_CUDA_ERROR()                                                \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUDA kernel
__global__ void doPreRotation(const var_t *xp1, var_t *yp, const var_t *t,
                              int N4, int shift, int stride, int N2,
                              var_t sine) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N4) {
    const var_t *xp1_i = xp1 + i * 2 * stride;
    const var_t *xp2_i = xp1 + stride * (N2 - 1) - i * 2 * stride;

    var_t yr, yi;
    yr = -S_MUL(*xp2_i, t[i << shift]) + S_MUL(*xp1_i, t[(N4 - i) << shift]);
    yi = -S_MUL(*xp2_i, t[(N4 - i) << shift]) - S_MUL(*xp1_i, t[i << shift]);

    yp[i * 2] = yr - S_MUL(yi, sine);
    yp[i * 2 + 1] = yi + S_MUL(yr, sine);
  }
}

void preRotateWithCuda(const var_t *host_xp1, var_t *host_yp,
                       const var_t *host_t, int N, int shift, int stride,
                       var_t sine) {
  int N2 = N >> 1;
  int N4 = N >> 2;
  var_t *dev_xp1;
  var_t *dev_yp;
  var_t *dev_t;

  cudaMalloc((void **)&dev_xp1, N4 * 2 * stride * sizeof(var_t));
  cudaMalloc((void **)&dev_yp, N4 * 2 * sizeof(var_t));
  cudaMalloc((void **)&dev_t, (N4 << shift) * sizeof(var_t));

  cudaMemcpy(dev_xp1, host_xp1, N4 * 2 * stride * sizeof(var_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_t, host_t, (N4 << shift) * sizeof(var_t),
             cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (N4 + blockSize - 1) / blockSize;

  doPreRotation<<<numBlocks, blockSize>>>(dev_xp1, dev_yp, dev_t, N4, shift,
                                          stride, N2, sine);

  cudaDeviceSynchronize();
  cudaMemcpy(host_yp, dev_yp, N4 * 2 * sizeof(var_t), cudaMemcpyDeviceToHost);

  cudaFree(dev_xp1);
  cudaFree(dev_yp);
  cudaFree(dev_t);
}

__global__ void postAndMirrorKernel(var_t *d_out, const var_t *t,
                                    const var_t *window, int N2, int N4,
                                    int shift, var_t sine, int overlap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Handle post-rotation part
  if (idx < (N4 + 1) >> 1) {
    var_t re, im, yr, yi;
    var_t t0, t1;

    // Calculate left pointer position
    var_t *yp0 = d_out + (overlap >> 1) + 2 * idx;
    // Calculate right pointer position
    var_t *yp1 = d_out + (overlap >> 1) + N2 - 2 - 2 * idx;

    // Process the first pair of values
    re = yp0[0];
    im = yp0[1];
    t0 = t[idx << shift];
    t1 = t[(N4 - idx) << shift];
    yr = S_MUL(re, t0) - S_MUL(im, t1);
    yi = S_MUL(im, t0) + S_MUL(re, t1);

    // Save the first pair of results
    var_t yr1 = yr;
    var_t yi1 = yi;

    // Process the second pair of values
    re = yp1[0];
    im = yp1[1];
    t0 = t[(N4 - idx - 1) << shift];
    t1 = t[(idx + 1) << shift];
    yr = S_MUL(re, t0) - S_MUL(im, t1);
    yi = S_MUL(im, t0) + S_MUL(re, t1);

    // Write results in the same order as the CPU version
    yp0[0] = -(yr1 - S_MUL(yi1, sine)); // Left real
    yp1[1] = yi1 + S_MUL(yr1, sine);    // Right imag
    yp1[0] = -(yr - S_MUL(yi, sine));   // Right real
    yp0[1] = yi + S_MUL(yr, sine);      // Left imag
  }

  // sync
  __syncthreads();

  // Handle mirror part
  // Use a different index for the mirror operation to ensure all elements are
  // processed
  int mirror_idx = idx;
  if (mirror_idx < overlap / 2) {
    var_t x1, x2;
    var_t *xp1 = d_out + overlap - 1 - mirror_idx;
    var_t *yp1 = d_out + mirror_idx;
    const var_t *wp1 = window + mirror_idx;
    const var_t *wp2 = window + overlap - 1 - mirror_idx;

    x1 = *xp1;
    x2 = *yp1;

    // Use temporary variables to avoid writing order issues
    var_t temp1 = S_MUL(*wp2, x2) - S_MUL(*wp1, x1);
    var_t temp2 = S_MUL(*wp1, x2) + S_MUL(*wp2, x1);

    *yp1 = temp1;
    *xp1 = temp2;
  }
}

__global__ void postAndMirrorKernelFused(var_t *d_out_ch0, var_t *d_out_ch1,
                                         const var_t *t, const var_t *window,
                                         int N2, int N4, int shift, var_t sine,
                                         int overlap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Handle post-rotation part for both channels
  if (idx < (N4 + 1) >> 1) {
    // Channel 0 processing
    {
      var_t re, im, yr, yi;
      var_t t0, t1;

      // Calculate left pointer position for channel 0
      var_t *yp0 = d_out_ch0 + (overlap >> 1) + 2 * idx;
      // Calculate right pointer position for channel 0
      var_t *yp1 = d_out_ch0 + (overlap >> 1) + N2 - 2 - 2 * idx;

      // Process the first pair of values
      re = yp0[0];
      im = yp0[1];
      t0 = t[idx << shift];
      t1 = t[(N4 - idx) << shift];
      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      // Save the first pair of results
      var_t yr1 = yr;
      var_t yi1 = yi;

      // Process the second pair of values
      re = yp1[0];
      im = yp1[1];
      t0 = t[(N4 - idx - 1) << shift];
      t1 = t[(idx + 1) << shift];
      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      // Write results in the same order as the CPU version
      yp0[0] = -(yr1 - S_MUL(yi1, sine)); // Left real
      yp1[1] = yi1 + S_MUL(yr1, sine);    // Right imag
      yp1[0] = -(yr - S_MUL(yi, sine));   // Right real
      yp0[1] = yi + S_MUL(yr, sine);      // Left imag
    }

    // Channel 1 processing
    {
      var_t re, im, yr, yi;
      var_t t0, t1;

      // Calculate left pointer position for channel 1
      var_t *yp0 = d_out_ch1 + (overlap >> 1) + 2 * idx;
      // Calculate right pointer position for channel 1
      var_t *yp1 = d_out_ch1 + (overlap >> 1) + N2 - 2 - 2 * idx;

      // Process the first pair of values
      re = yp0[0];
      im = yp0[1];
      t0 = t[idx << shift];
      t1 = t[(N4 - idx) << shift];
      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      // Save the first pair of results
      var_t yr1 = yr;
      var_t yi1 = yi;

      // Process the second pair of values
      re = yp1[0];
      im = yp1[1];
      t0 = t[(N4 - idx - 1) << shift];
      t1 = t[(idx + 1) << shift];
      yr = S_MUL(re, t0) - S_MUL(im, t1);
      yi = S_MUL(im, t0) + S_MUL(re, t1);

      // Write results in the same order as the CPU version
      yp0[0] = -(yr1 - S_MUL(yi1, sine)); // Left real
      yp1[1] = yi1 + S_MUL(yr1, sine);    // Right imag
      yp1[0] = -(yr - S_MUL(yi, sine));   // Right real
      yp0[1] = yi + S_MUL(yr, sine);      // Left imag
    }
  }

  // sync threads before mirror operation
  __syncthreads();

  // Handle mirror part for both channels
  int mirror_idx = idx;
  if (mirror_idx < overlap / 2) {
    // Channel 0 mirror
    {
      var_t x1, x2;
      var_t *xp1 = d_out_ch0 + overlap - 1 - mirror_idx;
      var_t *yp1 = d_out_ch0 + mirror_idx;
      const var_t *wp1 = window + mirror_idx;
      const var_t *wp2 = window + overlap - 1 - mirror_idx;

      x1 = *xp1;
      x2 = *yp1;

      // Use temporary variables to avoid writing order issues
      var_t temp1 = S_MUL(*wp2, x2) - S_MUL(*wp1, x1);
      var_t temp2 = S_MUL(*wp1, x2) + S_MUL(*wp2, x1);

      *yp1 = temp1;
      *xp1 = temp2;
    }

    // Channel 1 mirror
    {
      var_t x1, x2;
      var_t *xp1 = d_out_ch1 + overlap - 1 - mirror_idx;
      var_t *yp1 = d_out_ch1 + mirror_idx;
      const var_t *wp1 = window + mirror_idx;
      const var_t *wp2 = window + overlap - 1 - mirror_idx;

      x1 = *xp1;
      x2 = *yp1;

      // Use temporary variables to avoid writing order issues
      var_t temp1 = S_MUL(*wp2, x2) - S_MUL(*wp1, x1);
      var_t temp2 = S_MUL(*wp1, x2) + S_MUL(*wp2, x1);

      *yp1 = temp1;
      *xp1 = temp2;
    }
  }
}

__global__ void doPreRotationFused(const var_t *xp1_ch0, const var_t *xp1_ch1,
                                   var_t *yp_ch0, var_t *yp_ch1, const var_t *t,
                                   int N4, int shift, int stride, int N2,
                                   var_t sine) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N4) {
    // Process channel 0
    const var_t *xp1_i_ch0 = xp1_ch0 + i * 2 * stride;
    const var_t *xp2_i_ch0 = xp1_ch0 + stride * (N2 - 1) - i * 2 * stride;

    var_t yr_ch0, yi_ch0;
    yr_ch0 = -S_MUL(*xp2_i_ch0, t[i << shift]) +
             S_MUL(*xp1_i_ch0, t[(N4 - i) << shift]);
    yi_ch0 = -S_MUL(*xp2_i_ch0, t[(N4 - i) << shift]) -
             S_MUL(*xp1_i_ch0, t[i << shift]);

    yp_ch0[i * 2] = yr_ch0 - S_MUL(yi_ch0, sine);
    yp_ch0[i * 2 + 1] = yi_ch0 + S_MUL(yr_ch0, sine);

    // Process channel 1
    const var_t *xp1_i_ch1 = xp1_ch1 + i * 2 * stride;
    const var_t *xp2_i_ch1 = xp1_ch1 + stride * (N2 - 1) - i * 2 * stride;

    var_t yr_ch1, yi_ch1;
    yr_ch1 = -S_MUL(*xp2_i_ch1, t[i << shift]) +
             S_MUL(*xp1_i_ch1, t[(N4 - i) << shift]);
    yi_ch1 = -S_MUL(*xp2_i_ch1, t[(N4 - i) << shift]) -
             S_MUL(*xp1_i_ch1, t[i << shift]);

    yp_ch1[i * 2] = yr_ch1 - S_MUL(yi_ch1, sine);
    yp_ch1[i * 2 + 1] = yi_ch1 + S_MUL(yr_ch1, sine);
  }
}

void processMDCTCuda(const var_t *input, var_t *output, const var_t *trig,
                     int N, int shift, int stride, var_t sine, int overlap,
                     const var_t *window) {
  int N2 = N >> 1;
  int N4 = N >> 2;

  // Device pointers and memory allocation
  var_t *dev_input, *dev_output, *dev_t, *dev_window, *dev_f2;
  size_t size_input = N4 * 2 * stride * sizeof(var_t);
  size_t size_output = (N2 + overlap) * sizeof(var_t);
  size_t size_fft = N4 * 2 * sizeof(var_t);
  size_t size_trig = (N4 << shift) * sizeof(var_t);
  size_t size_window = overlap * sizeof(var_t);

  // Allocate and copy memory
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_input, size_input));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_output, size_output));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_t, size_trig));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_window, size_window));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_f2, size_fft));

  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_output, output, size_output, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_input, input, size_input, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_t, trig, size_trig, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_window, window, size_window, cudaMemcpyHostToDevice));

  // Pre-rotation
  int blockSize = 256;
  int numBlocks = (N4 + blockSize - 1) / blockSize;
  doPreRotation<<<numBlocks, blockSize>>>(dev_input, dev_f2, dev_t, N4, shift,
                                          stride, N2, sine);
  cudaDeviceSynchronize();

  // ifft
  cuda_fft_state *state = cuda_fft_alloc(N4, shift);
  if (!state) {
    fprintf(stderr, "Failed to allocate FFT state\n");
    exit(EXIT_FAILURE);
  }

  var_t *output_offset = dev_output + (overlap >> 1);
  int result =
      cuda_fft_execute(state, (const float *)dev_f2, (const float *)dev_f2,
                       (float *)output_offset, (float *)output_offset);

  if (result != 0) {
    fprintf(stderr, "FFT execution failed with error %d\n", result);
    cuda_fft_free(state);
    exit(EXIT_FAILURE);
  }

  CHECK_LAST_CUDA_ERROR(); // Check for errors after FFT execution
  cudaDeviceSynchronize(); // Ensure all operations are complete

  // Fused post-rotation and mirror kernel
  // Calculate the maximum number of threads needed
  int max_elements = max((N4 + 1) >> 1, overlap / 2);
  int numBlocksFused = (max_elements + blockSize - 1) / blockSize;
  postAndMirrorKernel<<<numBlocksFused, blockSize>>>(
      dev_output, dev_t, dev_window, N2, N4, shift, sine, overlap);
  CHECK_LAST_CUDA_ERROR();
  cudaDeviceSynchronize();

  // Copy final results
  CHECK_CUDA_ERROR(
      cudaMemcpy(output, dev_output, size_output, cudaMemcpyDeviceToHost));

  // Cleanup
  if (state)
    cuda_fft_free(state);
  cudaFree(dev_input);
  cudaFree(dev_output);
  cudaFree(dev_t);
  cudaFree(dev_window);
  cudaFree(dev_f2);
}

void printCudaVersion() {
  fprintf(stderr, "CUDA Compiled version: %d\n", __CUDACC_VER_MAJOR__);

  int runtime_ver;
  cudaRuntimeGetVersion(&runtime_ver);
  fprintf(stderr, "CUDA Runtime version: %d\n", runtime_ver);

  int driver_ver;
  cudaDriverGetVersion(&driver_ver);
  fprintf(stderr, "CUDA Driver version: %d\n", driver_ver);
}

#include <unordered_map>
static std::unordered_map<int, var_t *> dev_buf;
static std::unordered_map<int, cuda_fft_state *> fft_buf;

// Create MDCT CUDA state
mdct_cuda_state *mdct_cuda_create(int N, int shift, int stride, int overlap) {
  mdct_cuda_state *state = (mdct_cuda_state *)malloc(sizeof(mdct_cuda_state));
  if (!state)
    return nullptr;

  // Initialize configuration
  state->N = N;
  state->N2 = N >> 1;
  state->N4 = N >> 2;
  state->shift = shift;
  state->stride = stride;
  state->overlap = overlap;
  state->initialized = false;

  // Calculate buffer sizes
  state->size_input = state->N4 * 2 * stride * sizeof(var_t);
  state->size_output = (state->N2 + overlap) * sizeof(var_t);
  state->size_fft = state->N4 * 2 * sizeof(var_t);
  state->size_trig = (state->N4 << shift) * sizeof(var_t);
  state->size_window = overlap * sizeof(var_t);

  // Allocate device memory
  size_t total_size = state->size_input * 2 + state->size_output * 2 +
                      state->size_trig + state->size_window +
                      state->size_fft * 4;

  var_t *dev_buf;
  if (cudaMalloc(&dev_buf, total_size) != cudaSuccess) {
    free(state);
    return nullptr;
  }

  // Assign buffer pointers
  state->dev_input = dev_buf;
  state->dev_output = (var_t *)((char *)state->dev_input + state->size_input);
  state->dev_input1 = (var_t *)((char *)state->dev_output + state->size_output);
  state->dev_output1 = (var_t *)((char *)state->dev_input1 + state->size_input);
  state->dev_t = (var_t *)((char *)state->dev_output1 + state->size_output);
  state->dev_window = (var_t *)((char *)state->dev_t + state->size_trig);
  state->dev_f0 = (var_t *)((char *)state->dev_window + state->size_window);
  state->dev_f1 = (var_t *)((char *)state->dev_f0 + state->size_fft);
  state->dev_fft_output = (var_t *)((char *)state->dev_f1 + state->size_fft);

  // Create FFT plan
  if (cufftPlan1d(&state->plan, state->N4, CUFFT_C2C, 2) != CUFFT_SUCCESS) {
    cudaFree(dev_buf);
    free(state);
    return nullptr;
  }

  state->initialized = true;
  return state;
}

// Destroy MDCT CUDA state
void mdct_cuda_destroy(mdct_cuda_state *state) {
  if (state) {
    if (state->initialized) {
      cudaFree(
          state->dev_input); // Free all device memory (allocated as one block)
      cufftDestroy(state->plan);
    }
    free(state);
  }
}

// Process MDCT using persistent state
void mdct_cuda_process(mdct_cuda_state *state, const var_t *input[2],
                       var_t *output[2], const var_t *trig, const var_t *window,
                       var_t sine) {
  if (!state || !state->initialized)
    return;

  // Copy input data to device
  cudaMemcpy(state->dev_input, input[0], state->size_input,
             cudaMemcpyHostToDevice);
  cudaMemcpy(state->dev_input1, input[1], state->size_input,
             cudaMemcpyHostToDevice);
  cudaMemcpy(state->dev_output, output[0], state->size_output,
             cudaMemcpyHostToDevice);
  cudaMemcpy(state->dev_output1, output[1], state->size_output,
             cudaMemcpyHostToDevice);
  cudaMemcpy(state->dev_t, trig, state->size_trig, cudaMemcpyHostToDevice);
  cudaMemcpy(state->dev_window, window, state->size_window,
             cudaMemcpyHostToDevice);

  // Pre-rotation
  int blockSize = 256;
  int numBlocks = (state->N4 + blockSize - 1) / blockSize;

  doPreRotationFused<<<numBlocks, blockSize>>>(
      state->dev_input, state->dev_input1, state->dev_f0, state->dev_f1,
      state->dev_t, state->N4, state->shift, state->stride, state->N2, sine);

  // Execute FFT
  cuda_fft_state *state_fft = cuda_fft_alloc(state->N4, state->shift);
  if (!state_fft) {
    fprintf(stderr, "Failed to allocate FFT state\n");
    exit(EXIT_FAILURE);
  }

  var_t *c0_output_offset = state->dev_output + (state->overlap >> 1);
  var_t *c1_output_offset = state->dev_output1 + (state->overlap >> 1);
  int result = cuda_fft_execute(
      state_fft, (const float *)state->dev_f0, (const float *)state->dev_f1,
      (float *)c0_output_offset, (float *)c1_output_offset);

  if (result != 0) {
    fprintf(stderr, "FFT execution failed with error %d\n", result);
    cuda_fft_free(state_fft);
    exit(EXIT_FAILURE);
  }

  // Post-rotation and mirror
  int max_elements = max((state->N4 + 1) >> 1, state->overlap / 2);
  int numBlocksFused = (max_elements + blockSize - 1) / blockSize;
  postAndMirrorKernelFused<<<numBlocksFused, blockSize>>>(
      state->dev_output, state->dev_output1, state->dev_t, state->dev_window,
      state->N2, state->N4, state->shift, sine, state->overlap);

  // Copy results back to host
  cudaMemcpy(output[0], state->dev_output, state->size_output,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(output[1], state->dev_output1, state->size_output,
             cudaMemcpyDeviceToHost);
}

// Update the original function to use the new state management
void processMDCTCudaB1C2(const var_t *input[2], var_t *output[2],
                         const var_t *trig, int N, int shift, int stride,
                         var_t sine, int overlap, const var_t *window) {
  static mdct_cuda_state *state = nullptr;

  // Create state if not exists
  if (!state) {
    state = mdct_cuda_create(N, shift, stride, overlap);
    if (!state) {
      printf("Failed to create MDCT CUDA state\n");
      return;
    }
  }

  // Process using persistent state
  mdct_cuda_process(state, input, output, trig, window, sine);
}

// Update cleanup function
void cleanupCudaBuffers() {
  // Add cleanup for static state if needed
  // Note: This function might need to be called explicitly at program end
  for (auto &it : dev_buf) {
    cudaFree(it.second);
  }
  dev_buf.clear();
  for (auto &it : fft_buf) {
    cuda_fft_free(it.second);
  }
  fft_buf.clear();
}

// Performance test function
void performanceTest(int numIterations) {
  // Test parameters
  const int N = 2048; // FFT size
  const int shift = 1;
  const int stride = 1;
  const float sine = 0.0f;
  const int overlap = 256;

  // Allocate host memory
  const int N2 = N >> 1;
  const int N4 = N >> 2;
  const var_t *input_const[2];
  var_t *input[2], *output[2], *trig, *window;

  input[0] = new var_t[N4 * 2 * stride];
  input[1] = new var_t[N4 * 2 * stride];
  output[0] = new var_t[N2 + overlap];
  output[1] = new var_t[N2 + overlap];
  trig = new var_t[N4 << shift];
  window = new var_t[overlap];

  input_const[0] = input[0];
  input_const[1] = input[1];

  // Initialize test data
  for (int i = 0; i < N4 * 2 * stride; i++) {
    input[0][i] = (var_t)rand() / RAND_MAX;
    input[1][i] = (var_t)rand() / RAND_MAX;
  }
  for (int i = 0; i < N4 << shift; i++) {
    trig[i] = (var_t)rand() / RAND_MAX;
  }
  for (int i = 0; i < overlap; i++) {
    window[i] = (var_t)rand() / RAND_MAX;
  }

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup run
  processMDCTCudaB1C2(input_const, output, trig, N, shift, stride, sine,
                      overlap, window);
  cudaDeviceSynchronize();

  // Performance test
  float totalTime = 0.0f;
  float minTime = FLT_MAX;
  float maxTime = 0.0f;

  printf("\nRunning performance test with %d iterations...\n", numIterations);

  for (int i = 0; i < numIterations; i++) {
    cudaEventRecord(start);

    processMDCTCudaB1C2(input_const, output, trig, N, shift, stride, sine,
                        overlap, window);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    totalTime += milliseconds;
    minTime = min(minTime, milliseconds);
    maxTime = max(maxTime, milliseconds);

    if ((i + 1) % 10 == 0) {
      printf("Completed %d iterations...\n", i + 1);
    }
  }

  // Print performance statistics
  float avgTime = totalTime / numIterations;
  printf("\nPerformance Statistics (over %d iterations):\n", numIterations);
  printf("Average Time: %.4f ms\n", avgTime);
  printf("Min Time:     %.4f ms\n", minTime);
  printf("Max Time:     %.4f ms\n", maxTime);
  printf("Total Time:   %.4f ms\n", totalTime);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  delete[] input[0];
  delete[] input[1];
  delete[] output[0];
  delete[] output[1];
  delete[] trig;
  delete[] window;
}
