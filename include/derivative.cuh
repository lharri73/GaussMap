#include "gaussMap.hpp"

// calculates the first order approximation of the given array
// using the finite difference
__global__
void calcDerivative(float* f, array_info* fInfo, float *fprime, array_info *fPrimeInfo);
