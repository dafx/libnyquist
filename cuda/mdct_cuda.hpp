#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef var_t
#define var_t float
#endif

// Remove old function declarations
void mdctBackwardWithCuda(const var_t *in, var_t *out, 
                         const var_t *trig, const var_t *window,
                         int N, int overlap, int shift, int stride);

void printCudaVersion();

#ifdef __cplusplus
}
#endif
