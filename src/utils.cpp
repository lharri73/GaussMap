#include "utils.hpp"
/* All cuda functions return a cudaError_t, this function makes sure that it is
 * success. If not, it throws a runtime error. This is usefull for debugging 
 * rather than getting a segmentation fault */
void checkCudaError(cudaError_t error){
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
