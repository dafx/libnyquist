#include "mdct_cuda.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#ifndef S_MUL
#define S_MUL(a, b) ((a) * (b))
#endif

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_LAST_CUDA_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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
  int blockSize = 256; // Number of threads per block
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

void printCudaVersion() {
    fprintf(stderr, "CUDA Compiled version: %d\n", __CUDACC_VER_MAJOR__);

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    fprintf(stderr, "CUDA Runtime version: %d\n", runtime_ver);

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    fprintf(stderr, "CUDA Driver version: %d\n", driver_ver);
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

    // Debug: Check post-rotation output
    var_t *temp_post = (var_t *)malloc((N2 + overlap) * sizeof(var_t));
    CHECK_CUDA_ERROR(cudaMemcpy(temp_post, d_out, (N2 + overlap) * sizeof(var_t), cudaMemcpyDeviceToHost));
    printf("First 4 output values after post-rotation: %f, %f, %f, %f\n", 
           temp_post[overlap>>1], temp_post[(overlap>>1)+1], temp_post[(overlap>>1)+2], temp_post[(overlap>>1)+ 3]);
    free(temp_post);

    
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



void processMDCTSeparate(const var_t *input, var_t *output, const var_t *trig, int N, int shift, int stride, var_t sine, int overlap, const var_t *window) {
    int N2 = N >> 1;
    int N4 = N >> 2;
    var_t *f2 = (var_t *)malloc(N4 * 2 * sizeof(var_t));

    // pre-rotation
    preRotateWithCuda(input, f2, trig, N, shift, stride, sine);
    
    // ifft
    cuda_fft_state *state = cuda_fft_alloc(N4, shift);
    cuda_fft_execute(state, f2, output + (overlap >> 1));
    cuda_fft_free(state);

    // post-rotation and mirror
    postAndMirrorWithCuda(output, trig, N2, N4, shift, stride, sine, overlap, window);
    free(f2);
}

void processMDCTCuda(const var_t *input, var_t *output, const var_t *trig, int N, 
                     int shift, int stride, var_t sine, int overlap, const var_t *window) {
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

    // make sure to copy output to device !!!
    CHECK_CUDA_ERROR(cudaMemcpy(dev_output, output, size_output, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_input, input, size_input, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_t, trig, size_trig, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_window, window, size_window, cudaMemcpyHostToDevice));

    // Pre-rotation
    int blockSize = 256;
    int numBlocks = (N4 + blockSize - 1) / blockSize;
    doPreRotation<<<numBlocks, blockSize>>>(dev_input, dev_f2, dev_t, N4, shift, stride, N2, sine);
    cudaDeviceSynchronize();
    
    // ifft
    cuda_fft_state *state = cuda_fft_alloc(N4, shift);
    if (!state) {
        fprintf(stderr, "Failed to allocate FFT state\n");
        exit(EXIT_FAILURE);
    }

    var_t *output_offset = dev_output + (overlap >> 1);
    cufftResult result = cufftExecC2C(state->plan,
                                      (cufftComplex *)dev_f2,
                                      (cufftComplex *)output_offset,
                                      CUFFT_INVERSE);
    CHECK_LAST_CUDA_ERROR(); // Check for errors after FFT execution
    cudaDeviceSynchronize(); // Ensure all operations are complete

    // post-rotation
    int numElementsRotation = (N4 + 1) >> 1;
    int numBlocksRotation = (numElementsRotation + blockSize - 1) / blockSize;
    postRotationKernel<<<numBlocksRotation, blockSize>>>(dev_output, dev_t, 
                                                        N2, N4, shift, sine, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    
    // mirror
    int numElementsMirror = overlap / 2;
    int numBlocksMirror = (numElementsMirror + blockSize - 1) / blockSize;
    mirrorKernel<<<numBlocksMirror, blockSize>>>(dev_output, dev_window, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    // Copy final results and print
    CHECK_CUDA_ERROR(cudaMemcpy(output, dev_output, size_output, cudaMemcpyDeviceToHost));

    // Cleanup
    if (state) cuda_fft_free(state);
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_t);
    cudaFree(dev_window);
    cudaFree(dev_f2);
}

#include <unordered_map>
static std::unordered_map<int, var_t *> dev_buf;
static std::unordered_map<int, cuda_fft_state *> fft_buf;

void processMDCTCudaB1C2(const var_t *input[2], var_t *output[2], const var_t *trig, int N,
                         int shift, int stride, var_t sine, int overlap, const var_t *window)
{
    int N2 = N >> 1;
    int N4 = N >> 2;

    // Device pointers and memory allocation
    var_t *dev_input, *dev_output, *dev_t, *dev_window, *dev_f0, *dev_f1;
    var_t *dev_input1, *dev_output1;
    size_t size_input = N4 * 2 * stride * sizeof(var_t);
    size_t size_output = (N2 + overlap) * sizeof(var_t);
    size_t size_fft = N4 * 2 * sizeof(var_t);
    size_t size_trig = (N4 << shift) * sizeof(var_t);
    size_t size_window = overlap * sizeof(var_t);

    // Allocate and copy memory
    size_t total_dev_size = size_input * 2 + size_output * 2 + size_trig + size_window + size_fft * 4;
    var_t *dev_buf_ptr;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_buf_ptr, total_dev_size));
    dev_input = dev_buf_ptr;
    dev_output = (float*)((char *)dev_input + size_input);
    dev_input1 = (float*)((char *)dev_output + size_output);
    dev_output1 = (float*)((char *)dev_input1 + size_input);
    dev_t = (float*)((char *)dev_output1 + size_output);
    dev_window = (float*)((char *)dev_t + size_trig);
    dev_f0 = (float*)((char *)dev_window + size_window);
    dev_f1 = (float*)((char *)dev_f0 + size_fft);
    var_t *dev_fft_output = (float*)((char *)dev_f1 + size_fft);

    // if(dev_buf.find(total_dev_size) == dev_buf.end()) {
    //     CHECK_CUDA_ERROR(cudaMalloc((void **)&dev_buf_ptr, total_dev_size));
    //     dev_buf[total_dev_size] = dev_buf_ptr;
    // } else {
    //     dev_buf_ptr = dev_buf[total_dev_size];
    // }

    // make sure to copy output to device !!!
    CHECK_CUDA_ERROR(cudaMemcpy(dev_output, output[0], size_output, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_input, input[0], size_input, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_output1, output[1], size_output, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_input1, input[1], size_input, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_t, trig, size_trig, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_window, window, size_window, cudaMemcpyHostToDevice));

    // Pre-rotation
    int blockSize = 256;
    int numBlocks = (N4 + blockSize - 1) / blockSize;
    doPreRotation<<<numBlocks, blockSize>>>(dev_input, dev_f0, dev_t, N4, shift, stride, N2, sine);
    doPreRotation<<<numBlocks, blockSize>>>(dev_input1, dev_f1, dev_t, N4, shift, stride, N2, sine);
    cudaDeviceSynchronize();

    // ifft
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, N4, CUFFT_C2C, 2);
    if (result != CUFFT_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }

    // batch of 2
    result = cufftExecC2C(plan,
                          (cufftComplex *)dev_f0,
                          (cufftComplex *)dev_fft_output,
                          CUFFT_INVERSE);
    CHECK_LAST_CUDA_ERROR(); // Check for errors after FFT execution
    cudaDeviceSynchronize(); // Ensure all operations are complete
    cufftDestroy(plan);

    // ch 1
    var_t *c0_output_offset = dev_output + (overlap >> 1);
    var_t *c1_output_offset = dev_output1 + (overlap >> 1);
    CHECK_CUDA_ERROR(cudaMemcpy(c0_output_offset, dev_fft_output, size_fft, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(c1_output_offset, (char *)dev_fft_output + size_fft, size_fft, cudaMemcpyDeviceToDevice));
    CHECK_LAST_CUDA_ERROR(); // Check for errors after FFT execution
    cudaDeviceSynchronize(); // Ensure all operations are complete

    // post-rotation
    int numElementsRotation = (N4 + 1) >> 1;
    int numBlocksRotation = (numElementsRotation + blockSize - 1) / blockSize;
    postRotationKernel<<<numBlocksRotation, blockSize>>>(dev_output, dev_t,
                                                         N2, N4, shift, sine, overlap);
    CHECK_LAST_CUDA_ERROR();

    postRotationKernel<<<numBlocksRotation, blockSize>>>(dev_output1, dev_t,
                                                         N2, N4, shift, sine, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    // mirror
    int numElementsMirror = overlap / 2;
    int numBlocksMirror = (numElementsMirror + blockSize - 1) / blockSize;
    mirrorKernel<<<numBlocksMirror, blockSize>>>(dev_output, dev_window, overlap);
    CHECK_LAST_CUDA_ERROR();

    mirrorKernel<<<numBlocksMirror, blockSize>>>(dev_output1, dev_window, overlap);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    // Copy final results and print
    CHECK_CUDA_ERROR(cudaMemcpy(output[0], dev_output, size_output, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(output[1], dev_output1, size_output, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(dev_buf_ptr);
}

void cleanupCudaBuffers() {
    for (auto &it : dev_buf) {
        cudaFree(it.second);
    }
    dev_buf.clear();
    for (auto &it : fft_buf) {
        cuda_fft_free(it.second);
    }
    fft_buf.clear();
}
