#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gaussMap.hpp"
#include "utils.hpp"

// simple class used to hold the position information passed between cuda functions
class Position{
    public:
        __device__ Position(float x, float y);
        __device__ Position();
        __device__ void recalc();
        
        float x;
        float y;
        float radius;
};

// provides index of array given row, col, and array_info
__device__ size_t array_index(size_t row, size_t col, const array_info *info);

// calculates the difference (in meters) from a radar point to a cell
// returns a Position, including the radius and elementwise difference
__device__ Position indexDiff(size_t row, size_t col, const RadarData_t *radarData, 
                            size_t radarPointIdx, const array_info *radarInfo, 
                            const array_info *mapInfo, const array_rel *mapRel);

// given an index in the heatmap, calculates the position in the real world
// relative to the origin of the heatmap (center of the map)
__device__ Position index_to_position(size_t row, size_t col, 
                                const array_info *info, const array_rel *relation);

// calculates the probability distribution function given x, standard deviation, and mean
// (in this case, x=radius)
__device__ float calcPdf(float stdDev, float mean, float radius);

// kernel for radar points. calculates the pdf of each cell for each radar point and adds it
// to the value already in the heat map.
__global__ void radarPointKernel(mapType_t* gaussMap, const RadarData_t *radarData, 
                                const array_info* mapInfo, const array_rel* mapRel, 
                                const array_info* radarInfo, const distInfo_t* distributionInfo, radarId_t *radarIds);

__global__ void calcMaxKernel(uint8_t *isMax, const float* array, const array_info *mapInfo);

__device__ float calcMean(size_t col, const int16_t* radars, const RadarData_t *radarData, const array_info *radarInfo);

__global__ void aggregateMax(const mapType_t *array, const array_info *mapInfo,
                             const array_rel *mapRel, const maxVal_t *isMax, 
                             float* ret, const radarId_t *radarIds,
                             const array_info* maxInfo, float minCutoff,
                             const RadarData_t *radarData, const array_info *radarInfo);

__global__ void associateCameraKernel(const RadarData_t *radarData, const array_info *radarInfo,
                           const float* camData, const array_info *camInfo,
                           float* results, const array_info *resultInfo, float* tmp);
