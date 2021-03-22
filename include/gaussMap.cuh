#pragma once
#include <cuda_fp16.h>
#include "gaussMap.hpp"
#include "cudaUtils.hpp"

// Defines a set of gaurds to prevent overflows our out of 
// bounds errors. Useful for debugging 
#define CUDA_ASSERT_NEG(var, error) error_check(var < 0.0f, __FILE__, __LINE__, error, var);
#define CUDA_ASSERT_POS(var, error) error_check(var > 0.0f, __FILE__, __LINE__, error, var);
#define CUDA_ASSERT_POS_E(var, error) error_check(var >= 0.0f, __FILE__, __LINE__, error, var);
#define CUDA_ASSERT_GT(var1, var2, error) error_check(var1 >= var2, __FILE__, __LINE__, error, var1, var2);
#define CUDA_ASSERT_LT_E(var1, var2, error) error_check(var1 <= var2, __FILE__, __LINE__, error, var1, var2);
#define CUDA_ASSERT_GT_LINE(var1, var2, error, line) error_check(var1 >= var2, __FILE__, line, error, var1, var2, true);

__device__ 
__forceinline__
void error_check(bool condition, const char* file, int line, const char* error, float var, float var2=0, bool debug=false){
    if(!condition){
        if(!debug)
            printf("CUDA ERROR: %s\n\tGot %f (%f)\n", error, var, var2);
        else
            printf("CUDA ERROR at %s:%d :%s\n\tGot %f (%f)\n", file, line, error, var, var2);
        asm("trap;");   // inline ptx assembly to cause an illegal instruction
    }
}

// provides index of array given row, col, and array_info
__device__ __forceinline__
size_t array_index(size_t row, size_t col, const array_info *info, int line=0){
    if(line == 0){
        CUDA_ASSERT_GT(info->rows, row, "Index out of bounds: info->rows !> row");
        CUDA_ASSERT_GT(info->cols, col, "Index out of bounds: info->cols !> col");
    }else{
        CUDA_ASSERT_GT_LINE(info->rows, row, "Index out of bounds: info->rows !> row (filename wrong)",line);
        CUDA_ASSERT_GT_LINE(info->cols, col, "Index out of bounds: info->cols !> col (filename wrong)",line);
    }
    // helper function to find the array index
    return (row * info->cols) + col;
}

// calculates the difference (in meters) from a radar point to a cell
// returns a Position, including the radius and elementwise difference
__device__ Position_t indexDiff(
        size_t row, 
        size_t col, 
        const RadarData_t *radarData, 
        size_t radarPointIdx, 
        const array_info *radarInfo, 
        const array_info *mapInfo, 
        const array_rel *mapRel);

// given an index in the heatmap, calculates the position in the real world
// relative to the origin of the heatmap (center of the map)
__device__ Position_t index_to_position(
        size_t row, 
        size_t col, 
        const array_info *info, 
        const array_rel *relation);

// calculates the probability distribution function given x, standard deviation, and mean
// (in this case, x=radius)
__device__ float calcPdf(float stdDev, float mean, float radius);

// kernel for radar points. calculates the pdf of each cell for each radar point and adds it
// to the value already in the heat map.
__global__ void radarPointKernel(
        mapType_t* gaussMap, 
        const RadarData_t *radarData, 
        const array_info* mapInfo, 
        const array_rel* mapRel, 
        const array_info* radarInfo, 
        const distInfo_t* distributionInfo, 
        radarId_t *radarIds);

__global__ void calcMaxKernel(
        maxVal_t *isMax, 
        const float* array, 
        const array_info *mapInfo, 
        const radarId_t *radarIds,
        unsigned short windowSize);

__device__ float calcMean(
        size_t col,
        const int16_t* radars, 
        const RadarData_t *radarData, 
        const array_info *radarInfo);

__global__ void aggregateMax(
        const mapType_t *array, 
        const array_info *mapInfo,
        const array_rel *mapRel,
        const maxVal_t *isMax,
        float* ret,
        const array_info* maxInfo, 
        const RadarData_t *radarData, 
        const array_info *radarInfo, 
        const int *maximaLocs,
        const array_info *locsInfo);

__global__ void setSpaceMap(
        const RadarData_t *radarData,
        const array_info *radarInfo,
        const float* camData,
        const array_info *camInfo,
        float* spaceMap,
        const array_info *spaceMapInfo);

__global__ void associateCameraKernel(
        const RadarData_t *radarData, 
        const array_info *radarInfo,
        const float* camData, 
        const array_info *camInfo,
        float* results, 
        const array_info *resultInfo, 
        float* spaceMap,
        const array_info* spaceMapInfo);

__global__ void joinFeatures(
        const RadarData_t *radarData,
        const array_info *radarInfo,
        const float* camData,
        const array_info *camInfo,
        float* results,
        const array_info *resultInfo,
        float* spaceMap,
        const array_info *spaceMapInfo);

__global__ void singleElementResult(
        const float* radarData, 
        const array_info *radarInfo,
        const float* camData, 
        const array_info *camInfo,
        float* results, 
        const array_info *resultInfo);

__global__ void setRadarIdsKernel(radarId_t *array);