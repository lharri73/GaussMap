#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

void checkCudaError(cudaError_t error);