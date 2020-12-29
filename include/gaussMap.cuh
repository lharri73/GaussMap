#include <cuda_runtime.h>
#include <utility>
#include "gaussMap.hpp"
#include "utils.hpp"

typedef struct Position{
    float x;
    float y;
} position;

__device__ size_t array_index(size_t row, size_t col, array_info *info);

template <typename T>
__device__ float radiusFromPos(T x, T y);

__device__ position index_to_position(size_t row, size_t col, 
                                  array_info *info, array_rel *relation);

__device__ float calcPdf(float mean, float stdDev, float radius);

__global__ void radarPointKernel(mapType_t* gaussMap, RadarData_t *radarData, 
                                 array_info* mapInfo, array_rel* mapRel, 
                                 array_info* radarInfo, float* distributionInfo);
