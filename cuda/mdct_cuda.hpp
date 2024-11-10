#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef var_t
#define var_t float
#endif

#ifdef __CUDACC__
#include <cufft.h>

// Add the FFT-related type definition
typedef struct {
    cufftHandle plan;
    int nfft;
    int shift;
    cufftComplex *d_in;
    cufftComplex *d_out;
    int initialized;  // 添加初始化标志
} cuda_fft_state;

// Add function declarations for FFT operations
cuda_fft_state* cuda_fft_alloc(int nfft, int shift);
int cuda_fft_execute(cuda_fft_state *state, const float *input, float *output);
void cuda_fft_free(cuda_fft_state *state);
#endif

void doPreRotation(const float *input, float *output, int N);
void preRotateWithCuda(const var_t *host_xp1, var_t *host_yp,
                       const var_t *host_t, int N, int shift, int stride,
                       var_t sine);
void printCudaVersion();
void postAndMirrorWithCuda(var_t *out, const var_t *host_t, int N2, int N4, int shift, int stride, var_t sine, int overlap, const var_t *window);	

// New function declaration
void processMDCTCuda(const var_t *input, var_t *output, const var_t *trig, int N, int shift, int stride, var_t sine, int overlap, const var_t *window);


#ifdef __cplusplus
}
#endif
