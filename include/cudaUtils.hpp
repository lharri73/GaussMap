#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include "types.hpp"

void checkCudaError(cudaError_t error);
void safeCudaFree(void *ptr);

// These templated functions **must** be in the header because of the way 
// the compiler handles templated compilation with typedefs

template <typename T>
void safeCudaMalloc(T **ptr, size_t size);

template <typename T>
void safeCudaMemcpy2Device(T *dst, const T *src, size_t size);

template <typename T>
void safeCudaMemcpy2Host(T *dst, const T *src, size_t size);

template <typename T>
void safeCudaMemset(T* ptr, int value, size_t size);

#include "cudaUtils.cpp"