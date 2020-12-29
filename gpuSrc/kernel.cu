#include <cuda_runtime.h>
#include <sstream>
#include "gaussMap.hpp"
#include "utils.hpp"

__device__
size_t array_index(size_t row, size_t col, array_info info){
    return row * info.rows + col;
}

__global__ 
void radarPointKernel(short* gaussMap, RadarData_t *radarData, dim3 radarSize){

}

void GaussMap::calcRadarMap(){
    dim3 arraySize(numPoints, radarFeatures);

    // dispatch the kernel with `numPoints` threads
    radarPointKernel<<<1,numPoints>>>(
        array,
        radarData,
        arraySize
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "radarPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::string(ss.str());
    }

    // wait untill all threads sync
    cudaDeviceSynchronize();
}
