#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include "types.hpp"

void checkCudaError(cudaError_t error);
void safeCudaFree_macro(void *ptr, int line, const char* file);

// These templated functions **must** be in the header because of the way 
// the compiler handles templated compilation with typedefs

template <typename T>
void safeCudaMalloc_macro(T **ptr, size_t size, int line, const char* file);

template <typename T>
void safeCudaMemcpy2Device_macro(T *dst, const T *src, size_t size, int line, const char* file);

template <typename T>
void safeCudaMemcpy2Host_macro(T *dst, const T *src, size_t size, int line, const char* file);

template <typename T>
void safeCudaMemset_macro(T* ptr, int value, size_t size, int line, const char* file);

#include "cudaUtils.cpp"

#define safeCudaMalloc(ptr,size) safeCudaMalloc_macro(ptr,size,__LINE__,__FILE__)
#define safeCudaMemcpy2Device(dst,src,size) safeCudaMemcpy2Device_macro(dst,src,size,__LINE__,__FILE__)
#define safeCudaMemcpy2Host(dst,src,size) safeCudaMemcpy2Host_macro(dst,src,size,__LINE__,__FILE__)
#define safeCudaMemset(ptr,value,size) safeCudaMemset_macro(ptr,value,size,__LINE__,__FILE__)
#define safeCudaFree(ptr) safeCudaFree_macro(ptr,__LINE__,__FILE__)