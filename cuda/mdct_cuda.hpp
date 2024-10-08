#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

void doPreRotation(const float *input, float *output, int N);
void printCudaVersion();

#ifdef __cplusplus
}
#endif

