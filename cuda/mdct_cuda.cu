#include "mdct_cuda.hpp"

__global__ void doPreRotation(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float angle = M_PI / (2.0f * N) * (idx + 0.5f);
        output[idx] = input[idx] * cosf(angle) - input[N - 1 - idx] * sinf(angle);
    }
}
