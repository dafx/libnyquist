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


void postAndMirrorWithCuda(var_t *out, const var_t *host_t, int N2, int N4, int shift, int stride, var_t sine,int overlap, const var_t *window);	

#ifdef __cplusplus
}
#endif
