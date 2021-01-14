#include "utils.hpp"
/* All cuda functions return a cudaError_t, this function makes sure that it is
 * success. If not, it throws a runtime error. This is usefull for debugging 
 * rather than getting a segmentation fault */
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