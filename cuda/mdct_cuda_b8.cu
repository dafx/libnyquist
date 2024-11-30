#include "mdct_cuda.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <float.h>
#include <iostream>

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

#define CHECK_CUDA_ERROR(call)                                               \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                              \
    do                                                                       \
    {                                                                        \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// MDCT state management structure
typedef struct
{
    // FFT plan
    cufftHandle plan;
    cuda_fft_state *fft_state; // Persistent FFT state

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
} mdct_b8_cuda_state;

// MDCT state management functions
mdct_b8_cuda_state *mdct_b8_cuda_create(int N, int shift, int stride, int overlap);
void mdct_b8_cuda_destroy(mdct_b8_cuda_state *state);
void mdct_b8_cuda_process(mdct_b8_cuda_state *state, const var_t *input[2],
                          var_t *output[2], const var_t *trig, const var_t *window,
                          var_t sine);

__global__ void postAndMirrorKernelFused8(var_t *d_out_ch0, var_t *d_out_ch1,
                                         const var_t *t, const var_t *window,
                                         int N2, int N4, int shift, var_t sine,
                                         int overlap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle post-rotation part for both channels
    if (idx < (N4 + 1) >> 1)
    {
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
    if (mirror_idx < overlap / 2)
    {
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

__global__ void doPreRotationFused8(const var_t *xp1_ch0, const var_t *xp1_ch1,
                                   var_t *yp_ch0, var_t *yp_ch1, const var_t *t,
                                   int N4, int shift, int stride, int N2,
                                   var_t sine)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N4)
    {
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

// Create MDCT CUDA state
mdct_b8_cuda_state *mdct_b8_cuda_create(int N, int shift, int stride, int overlap)
{
    mdct_b8_cuda_state *state = (mdct_b8_cuda_state *)malloc(sizeof(mdct_b8_cuda_state));
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
    if (cudaMalloc(&dev_buf, total_size) != cudaSuccess)
    {
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
    if (cufftPlan1d(&state->plan, state->N4, CUFFT_C2C, 2) != CUFFT_SUCCESS)
    {
        cudaFree(dev_buf);
        free(state);
        return nullptr;
    }

    // Create FFT state
    state->fft_state = cuda_fft_alloc(state->N4, state->shift);
    if (!state->fft_state)
    {
        cufftDestroy(state->plan);
        cudaFree(dev_buf);
        free(state);
        return nullptr;
    }

    state->initialized = true;
    return state;
}

// Destroy MDCT CUDA state
void mdct_b8_cuda_destroy(mdct_cuda_state *state)
{
    if (state)
    {
        if (state->initialized)
        {
            cudaFree(
                state->dev_input); // Free all device memory (allocated as one block)
            cufftDestroy(state->plan);
            if (state->fft_state)
            {
                cuda_fft_free(state->fft_state);
            }
        }
        free(state);
    }
}

// Process MDCT using persistent state
void mdct_b8_cuda_process(mdct_b8_cuda_state *state, const var_t *input[2],
                       var_t *output[2], const var_t *trig, const var_t *window,
                       var_t sine)
{
    if (!state || !state->initialized)
        return;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEvent_t memAlloc_start, h2d_start, preproc_start, fft_start;
    cudaEvent_t fft_plan_start, fft_exec_start, fft_cleanup_start;
    cudaEvent_t postproc_start, d2h_start, cleanup_start;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&memAlloc_start);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&preproc_start);
    cudaEventCreate(&fft_start);
    cudaEventCreate(&fft_plan_start);
    cudaEventCreate(&fft_exec_start);
    cudaEventCreate(&fft_cleanup_start);
    cudaEventCreate(&postproc_start);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&cleanup_start);

    cudaEventRecord(start);
    cudaEventRecord(memAlloc_start);

    // Memory allocation timing would go here if any dynamic allocation was needed

    cudaEventRecord(h2d_start);
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

    cudaEventRecord(preproc_start);
    // Pre-rotation
    int blockSize = 256;
    int numBlocks = (state->N4 + blockSize - 1) / blockSize;

    doPreRotationFused8<<<numBlocks, blockSize>>>(
        state->dev_input, state->dev_input1, state->dev_f0, state->dev_f1,
        state->dev_t, state->N4, state->shift, state->stride, state->N2, sine);

    var_t *c0_output_offset = state->dev_output + (state->overlap >> 1);
    var_t *c1_output_offset = state->dev_output1 + (state->overlap >> 1);

    cudaEventRecord(fft_start);
    cudaEventRecord(fft_plan_start);
    // Execute FFT using the persistent FFT state
    cudaEventRecord(fft_exec_start);
    int result = cuda_fft_execute(
        state->fft_state, (const float *)state->dev_f0, (const float *)state->dev_f1,
        (float *)c0_output_offset, (float *)c1_output_offset);

    if (result != 0)
    {
        fprintf(stderr, "FFT execution failed with error %d\n", result);
        return;
    }

    cudaEventRecord(fft_cleanup_start);
    cudaEventRecord(postproc_start);

    // Post-rotation and mirror
    int max_elements = max((state->N4 + 1) >> 1, state->overlap / 2);
    int numBlocksFused = (max_elements + blockSize - 1) / blockSize;
    postAndMirrorKernelFused8<<<numBlocksFused, blockSize>>>(
        state->dev_output, state->dev_output1, state->dev_t, state->dev_window,
        state->N2, state->N4, state->shift, sine, state->overlap);

    cudaEventRecord(d2h_start);
    // Copy results back to host
    cudaMemcpy(output[0], state->dev_output, state->size_output,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(output[1], state->dev_output1, state->size_output,
               cudaMemcpyDeviceToHost);

    cudaEventRecord(cleanup_start);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate timing
    float total_time, memAlloc_time, h2d_time, preproc_time, fft_total_time;
    float fft_plan_time, fft_exec_time, fft_cleanup_time, postproc_time;
    float d2h_time, cleanup_time, other_time;

    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&memAlloc_time, memAlloc_start, h2d_start);
    cudaEventElapsedTime(&h2d_time, h2d_start, preproc_start);
    cudaEventElapsedTime(&preproc_time, preproc_start, fft_start);
    cudaEventElapsedTime(&fft_total_time, fft_start, postproc_start);
    cudaEventElapsedTime(&fft_plan_time, fft_plan_start, fft_exec_start);
    cudaEventElapsedTime(&fft_exec_time, fft_exec_start, fft_cleanup_start);
    cudaEventElapsedTime(&fft_cleanup_time, fft_cleanup_start, postproc_start);
    cudaEventElapsedTime(&postproc_time, postproc_start, d2h_start);
    cudaEventElapsedTime(&d2h_time, d2h_start, cleanup_start);
    cudaEventElapsedTime(&cleanup_time, cleanup_start, stop);

    float fft_overhead = fft_total_time - (fft_plan_time + fft_exec_time + fft_cleanup_time);
    other_time = total_time - (memAlloc_time + h2d_time + preproc_time + fft_total_time +
                               postproc_time + d2h_time + cleanup_time);

    // Print timing statistics
    printf("\nMDCT CUDA Timing Statistics:\n");
    printf("Total Time:                  %.3f ms (100.0%%)\n", total_time);
    printf("Memory Allocation:           %.3f ms (%5.1f%%)\n", memAlloc_time, 100.0f * memAlloc_time / total_time);
    printf("Host to Device Transfer:     %.3f ms (%5.1f%%)\n", h2d_time, 100.0f * h2d_time / total_time);
    printf("Pre-processing:              %.3f ms (%5.1f%%)\n", preproc_time, 100.0f * preproc_time / total_time);
    printf("IFFT Total:                  %.3f ms (%5.1f%%)\n", fft_total_time, 100.0f * fft_total_time / total_time);
    printf("  IFFT Plan:                 %.3f ms (%5.1f%%)\n", fft_plan_time, 100.0f * fft_plan_time / total_time);
    printf("  IFFT Execute:              %.3f ms (%5.1f%%)\n", fft_exec_time, 100.0f * fft_exec_time / total_time);
    printf("  IFFT Cleanup:              %.3f ms (%5.1f%%)\n", fft_cleanup_time, 100.0f * fft_cleanup_time / total_time);
    printf("  IFFT Overhead:             %.3f ms (%5.1f%%)\n", fft_overhead, 100.0f * fft_overhead / total_time);
    printf("Post-processing:             %.3f ms (%5.1f%%)\n", postproc_time, 100.0f * postproc_time / total_time);
    printf("Device to Host Transfer:     %.3f ms (%5.1f%%)\n", d2h_time, 100.0f * d2h_time / total_time);
    printf("Cleanup:                     %.3f ms (%5.1f%%)\n", cleanup_time, 100.0f * cleanup_time / total_time);
    printf("Other/Overhead:              %.3f ms (%5.1f%%)\n", other_time, 100.0f * other_time / total_time);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(memAlloc_start);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(preproc_start);
    cudaEventDestroy(fft_start);
    cudaEventDestroy(fft_plan_start);
    cudaEventDestroy(fft_exec_start);
    cudaEventDestroy(fft_cleanup_start);
    cudaEventDestroy(postproc_start);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(cleanup_start);
}

// Update the original function to use the new state management
void processMDCTCudaB8C2(const var_t *input[2], var_t *output[2],
                         const var_t *trig, int N, int shift, int stride,
                         var_t sine, int overlap, const var_t *window)
{
    static mdct_b8_cuda_state *state = nullptr;

    // Create state if not exists
    if (!state)
    {
        state = mdct_b8_cuda_create(N, shift, stride, overlap);
        if (!state)
        {
            printf("Failed to create MDCT CUDA state\n");
            return;
        }
    }

    // Process using persistent state
    mdct_b8_cuda_process(state, input, output, trig, window, sine);
}
