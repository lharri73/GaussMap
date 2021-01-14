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
void safeCudaMalloc(T **ptr, size_t size){
    cudaError_t error = cudaMalloc(ptr, size);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMalloc\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Device(T *dst, T *src, size_t size){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Device\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Host(T *dst, T *src, size_t size){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Host\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}