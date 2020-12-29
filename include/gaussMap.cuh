#include <cuda_runtime.h>
#include "gaussMap.hpp"
#include "utils.hpp"

__device__ size_t array_index(size_t row, size_t col, array_info *info);

template <typename T>
__device__ double radiusFromPos(T x, T y);

__device__ dim3 index_to_position(size_t row, size_t col, 
                                  array_info *info, array_rel *relation);

__global__ void radarPointKernel(mapType_t* gaussMap, RadarData_t *radarData, 
                                 array_info* mapInfo, array_rel* mapRel, 
                                 array_info* radarInfo);
