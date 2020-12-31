#include "derivative.cuh"
#include "gaussMap.cuh"

__global__
void calcDerivativeKernel(float* f, array_info *fInfo, float* fprime, array_info *fPrimeInfo){
    // https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences 
    // ^ 5th equation

}

float* GaussMap::calcDerivative(){
    primeInfo.rows = mapInfo.rows -2;
    primeInfo.cols = mapInfo.cols -2;
    primeInfo.elementSize = sizeof(float);

    checkCudaError(cudaMalloc(&primeInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMemcpy(primeInfo_cuda, &primeInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    if(mapInfo_cuda == nullptr)
        throw std::runtime_error("radar data must be added before the derivative can be calculated.");
    
    checkCudaError(cudaMalloc(&arrayPrime, sizeof(float) * primeInfo.rows * primeInfo.cols));

    // dispatch the kernel with a single thread per cell
    calcDerivativeKernel<<<primeInfo.rows, primeInfo.cols>>>(
        array,
        mapInfo_cuda,
        arrayPrime,
        primeInfo_cuda
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calcDerivativeKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

    cudaDeviceSynchronize();

    return nullptr;
}