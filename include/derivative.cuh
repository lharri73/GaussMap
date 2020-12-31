#include "gaussMap.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

// calculates the first order approximation of the given array
// using the finite difference
__global__
void calcDerivativeKernel(float* f, array_info* fInfo,
                    float *fprime, array_info *fPrimeInfo);
