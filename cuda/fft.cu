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
    
    // Create plan
    cufftResult result = cufftPlan1d(&state->plan, nfft, CUFFT_C2C, 1);
    if (result != CUFFT_SUCCESS) {
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

// Function to execute FFT
int cuda_fft_execute(cuda_fft_state *state, 
                    const float *input,
                    float *output) {
    if (!state || !state->initialized) {
        return -1;
    }
    
    // Check input
    if (!input || !output) {
        return -2;
    }
    
    cudaError_t error;
    
    // Copy input data
    error = cudaMemcpy(state->d_in, input,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return -3;
    }
    
    // Execute FFT
    cufftResult result = cufftExecC2C(state->plan,
                                     (cufftComplex *)state->d_in,
                                     (cufftComplex *)state->d_out,
                                     CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        return -4;
    }
    
    // Wait for GPU to complete
    cudaDeviceSynchronize();
    
    // Copy results
    error = cudaMemcpy(output, state->d_out,
                      state->nfft * sizeof(cufftComplex),
                      cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return -5;
    }
    
    return 0;
}