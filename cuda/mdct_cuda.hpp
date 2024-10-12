#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef var_t
#define var_t float
#endif

void doPreRotation(const float *input, float *output, int N);
void preRotateWithCuda(const var_t *host_xp1, var_t *host_yp,
                       const var_t *host_t, int N, int shift, int stride,
                       var_t sine);
void printCudaVersion();

#ifdef __cplusplus
}
#endif
