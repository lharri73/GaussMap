#ifndef NUSCENES
#include "ecocar_fusion/cudaUtils.hpp"
#else
#include "cudaUtils.hpp"
#endif

void checkCudaError(cudaError_t error){
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap internal error...\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

void safeCudaFree_macro(void *ptr, int line, const char* file){
    if(ptr != nullptr){
        cudaError_t error = cudaFree(ptr);
        if(error != cudaSuccess){
            std::stringstream ss;
            ss << "gaussMap:: Internal error during cudaFree at " << file << ":" << line << '\n';
            ss << "\tCUDA: " << error << '\n';
            throw std::runtime_error(ss.str());
        }
    }
}

size_t array_info::size(){
    return rows * cols * elementSize;
}
