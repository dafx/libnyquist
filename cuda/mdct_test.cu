#include "mdct_cuda.hpp"
#include <stdio.h>

int main() {
    // Print CUDA version
    printCudaVersion();
    
    // Run performance test
    printf("\nRunning performance test...\n");
    performanceTest(100);  // Run 100 iterations
    
    // Cleanup
    cleanupCudaBuffers();
    
    return 0;
}
