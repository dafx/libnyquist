#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef var_t
#define var_t float
#endif

#include <cufft.h>

// Add the FFT-related type definition
typedef struct {
  cufftHandle plan;
  int nfft;
  int shift;
  cufftComplex *d_in;
  cufftComplex *d_out;
  int initialized;
} cuda_fft_state;

// Add function declarations for FFT operations
cuda_fft_state *cuda_fft_alloc(int nfft, int shift);
int cuda_fft_execute(cuda_fft_state *state, const float *input_ch0, const float *input_ch1,
                    float *output_ch0, float *output_ch1);
void cuda_fft_free(cuda_fft_state *state);

// MDCT state management structure
typedef struct {
  // FFT plan
  cufftHandle plan;

  // Device buffers
  var_t *dev_input;      // Input buffer for channel 1
  var_t *dev_input1;     // Input buffer for channel 2
  var_t *dev_output;     // Output buffer for channel 1
  var_t *dev_output1;    // Output buffer for channel 2
  var_t *dev_t;          // Trig table
  var_t *dev_window;     // Window function
  var_t *dev_f0;         // FFT buffer for channel 1
  var_t *dev_f1;         // FFT buffer for channel 2
  var_t *dev_fft_output; // FFT output buffer

  // Configuration
  int N;       // FFT size
  int N2;      // N/2
  int N4;      // N/4
  int shift;   // Shift parameter
  int stride;  // Stride parameter
  int overlap; // Overlap size
  var_t sine;  // Sine parameter

  // Buffer sizes
  size_t size_input;
  size_t size_output;
  size_t size_trig;
  size_t size_window;
  size_t size_fft;

  // State
  int initialized;
} mdct_cuda_state;

// MDCT state management functions
mdct_cuda_state *mdct_cuda_create(int N, int shift, int stride, int overlap);
void mdct_cuda_destroy(mdct_cuda_state *state);
void mdct_cuda_process(mdct_cuda_state *state, const var_t *input[2],
                       var_t *output[2], const var_t *trig, const var_t *window,
                       var_t sine);

void doPreRotation(const float *input, float *output, int N);
void preRotateWithCuda(const var_t *host_xp1, var_t *host_yp,
                       const var_t *host_t, int N, int shift, int stride,
                       var_t sine);
void printCudaVersion();
void postAndMirrorWithCuda(var_t *out, const var_t *host_t, int N2, int N4,
                           int shift, int stride, var_t sine, int overlap,
                           const var_t *window);

// New function declaration
void processMDCTCuda(const var_t *input, var_t *output, const var_t *trig,
                     int N, int shift, int stride, var_t sine, int overlap,
                     const var_t *window);
void processMDCTCudaB1C2(const var_t *input[2], var_t *output[2],
                         const var_t *trig, int N, int shift, int stride,
                         var_t sine, int overlap, const var_t *window);

void cleanupCudaBuffers();

// Performance test function declaration
void performanceTest(int numIterations);

#ifdef __cplusplus
}
#endif
