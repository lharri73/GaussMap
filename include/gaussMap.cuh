#include <cuda_runtime.h>
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
__device__ size_t array_index(size_t row, size_t col, array_info *info);

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

__global__
void calcDerivativeKernel(float* f, array_info* fInfo,
                    float *fprime, array_info *fPrimeInfo);

__global__
void calcMaxKernel(uint8_t *isMax, float* arrayPrime, 
                   array_info *primeInfo, float* arrayPrimePrime, 
                   array_info *primePrimeInfo);