#include <cuda_runtime.h>
#include <utility>
#include "gaussMap.hpp"
#include "utils.hpp"
#include "Position.cuh"

// provides index of array given row, col, and array_info
__device__ size_t array_index(size_t row, size_t col, array_info *info);

// uses CUDA math api to calculate the radius given x and y
template <typename T>
__device__ float radiusFromPos(T x, T y);

// calculates the difference (in meters) from a radar point to a cell
// returns a Position, including the radius and elementwise difference
__device__ Position indexDiff(size_t row, size_t col, RadarData_t *radarData, 
                              size_t radarPointIdx, array_info *radarInfo, 
                              array_info *mapInfo, array_rel *mapRel);

// given an index in the heatmap, calculates the position in the real world
// relative to the origin of the heatmap (center of the map)
__device__ Position index_to_position(size_t row, size_t col, 
                                  array_info *info, array_rel *relation);

// calculates the probability distribution function given x, standard deviation, and mean
// (in this case, x=radius)
__device__ float calcPdf(float stdDev, float mean, float radius);

// kernel for radar points. calculates the pdf of each cell for each radar point and adds it
// to the value already in the heat map.
__global__ void radarPointKernel(mapType_t* gaussMap, RadarData_t *radarData, 
                                 array_info* mapInfo, array_rel* mapRel, 
                                 array_info* radarInfo, float* distributionInfo);
