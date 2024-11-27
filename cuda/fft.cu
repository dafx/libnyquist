#include <cufft.h>
#include <stdio.h>
#include "mdct_cuda.hpp"

// Initialization function
cuda_fft_state* cuda_fft_alloc(int nfft, int shift) {
    cuda_fft_state *state = (cuda_fft_state*)malloc(sizeof(cuda_fft_state));
    if (!state) {
        return NULL;
    }
    
    // Initialize all fields to 0
    memset(state, 0, sizeof(cuda_fft_state));
    
    state->nfft = nfft;
    state->shift = shift;
    
    // Create plan for 2 channels (batch = 2)
    cufftResult result = cufftPlan1d(&state->plan, nfft, CUFFT_C2C, 2);
    if (result != CUFFT_SUCCESS) {
        free(state);
        return NULL;
    }
    
    // Allocate device memory for both channels
    cudaError_t error;
    error = cudaMalloc((void**)&state->d_in, 2 * nfft * sizeof(cufftComplex));
    if (error != cudaSuccess) {
        cufftDestroy(state->plan);
        free(state);
        return NULL;
    }
    
    error = cudaMalloc((void**)&state->d_out, 2 * nfft * sizeof(cufftComplex));
    if (error != cudaSuccess) {
        cudaFree(state->d_in);
        cufftDestroy(state->plan);
        free(state);
        return NULL;
    }
    
    state->initialized = 1;
    return state;
}

// Safe release function
void cuda_fft_free(cuda_fft_state *state) {
    if (state) {
        if (state->initialized) {
            // First synchronize all CUDA operations
            cudaDeviceSynchronize();
            
            // Check and free device memory
            if (state->d_in) {
                cudaFree(state->d_in);
                state->d_in = NULL;
            }
            
            if (state->d_out) {
                cudaFree(state->d_out);
                state->d_out = NULL;
            }
            
            // Destroy plan
            if (state->plan) {
                cufftDestroy(state->plan);
            }
            
            state->initialized = 0;
        }
        
        // Finally, free the structure
        free(state);
    }
}

// Function to execute FFT for both channels
int cuda_fft_execute(cuda_fft_state *state, 
                    const float *input_ch0,
                    const float *input_ch1,
                    float *output_ch0,
                    float *output_ch1) {
    if (!state || !state->initialized) {
        return -1;
    }
    
    // Check inputs
    if (!input_ch0 || !input_ch1 || !output_ch0 || !output_ch1) {
        return -2;
    }
    
    cudaError_t error;
    
    // Copy input data for both channels
    error = cudaMemcpy(state->d_in, input_ch0,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return -3;
    }
    
    error = cudaMemcpy((cufftComplex*)state->d_in + state->nfft, input_ch1,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return -3;
    }
    
    // Execute FFT for both channels in one call
    cufftResult result = cufftExecC2C(state->plan,
                                     (cufftComplex*)state->d_in,
                                     (cufftComplex*)state->d_out,
                                     CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        return -4;
    }
    
    // Copy results back for both channels
    error = cudaMemcpy(output_ch0, state->d_out,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return -5;
    }
    
    error = cudaMemcpy(output_ch1, (cufftComplex*)state->d_out + state->nfft,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return -5;
    }
    
    return 0;
}