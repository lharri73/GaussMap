#include "cudaUtils.hpp"

void checkCudaError(cudaError_t error){
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap internal error...\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

void safeCudaFree(void *ptr){
    if(ptr != nullptr){
        cudaError_t error = cudaFree(ptr);
        if(error != cudaSuccess){
            std::stringstream ss;
            ss << "gaussMap:: Internal error during cudaFree\n";
            ss << "\tCUDA: " << error << '\n';
        }
    }
}

size_t array_info::size(){
    return rows * cols * elementSize;
}