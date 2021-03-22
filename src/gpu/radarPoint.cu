#include "gaussMap.cuh"
#include <math_constants.h>     // CUDART_PI_F

__device__ __forceinline__
float calcPdf(float stdDev, float mean, float radius){
    // calculate the pdf of the given radar point based on the radius
    CUDA_ASSERT_POS_E(radius, "Cannot calculate pdf for radius < 0");
    CUDA_ASSERT_POS(stdDev, "standard deviation cannot be <= 0");
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (radius - mean) / stdDev;
    float b = exp(-0.5f * a * a);
    return ((inv_sqrt_2pi / stdDev) * b);
    // float variance = pow(stdDev, 2);
    // float first = (1 / stdDev) * rsqrt(2 * CUDART_PI_F);
    // float exponent = -1 * pow(radius - mean, 2) / (2 * variance);
    // float second = exp(exponent);
    // return first*second;
}

__device__
Position_t indexDiff(size_t row, size_t col, const RadarData_t *radarData, size_t radarPointIdx, 
                   const array_info *radarInfo, const array_info *mapInfo, const array_rel *mapRel){
    // Calculate the position of the cell at (row,col) relative to the radar point at 
    // radarPointIdx
    Position_t pos = index_to_position(row, col, mapInfo, mapRel);
    
    float rPosx = radarData[array_index(radarPointIdx, 0, radarInfo)];
    float rPosy = radarData[array_index(radarPointIdx, 1, radarInfo)];

    Position_t difference;
    difference.x = pos.x - rPosx;
    difference.y = pos.y - rPosy;

    return difference;
}

__device__ 
Position_t index_to_position(size_t row, size_t col, const array_info *info, const array_rel *relation){
    // find the position from center of map given cell index
    float center_x = (float)(info->cols/2.0);
    float center_y = (float)(info->rows/2.0);
    float x_offset = (col - center_x);
    float y_offset = (row - center_y) * -1;     // flip the y axis so + is in the direction of travel

    Position_t ret;
    
    ret.x = x_offset / (float)relation->res;
    ret.y = y_offset / (float)relation->res;

    return ret;
}

__global__ 
void radarPointKernel(mapType_t* gaussMap, 
                      const RadarData_t *radarData, 
                      const array_info *mapInfo, 
                      const array_rel* mapRel, 
                      const array_info* radarInfo,
                      const distInfo_t* distributionInfo,
                      radarId_t *radarIds){
    // In this function, the radar point id is threadIdx.x
    union{
        radarId_t radData;
        unsigned long long int ulong;
    } un;

    float stdDev = (1-radarData[array_index(threadIdx.x, 4, radarInfo)] * distributionInfo->stdDev);
    
    for(size_t col = 0; col < mapInfo->cols; col++){
        // find where the cell is relative to the radar point
        Position_t diff = indexDiff(blockIdx.x, col, 
                                    radarData, threadIdx.x, 
                                    radarInfo, mapInfo, mapRel);
        
        float radius;
        radius = hypotf(diff.x,diff.y);
        // don't calculate the pdf of this cell if it's too far away
        if(radius > distributionInfo->distCutoff)
            continue;

        float pdfVal = calcPdf(stdDev, distributionInfo->mean, radius);
        CUDA_ASSERT_POS(pdfVal, "negative pdf value");

        atomicAdd(&gaussMap[array_index(blockIdx.x,col,mapInfo)], pdfVal);

        un.radData.radarId = threadIdx.x;
        un.radData.probability = pdfVal;
        atomicMax((unsigned long long int*)&radarIds[array_index(blockIdx.x, col, mapInfo)], un.ulong);

    }
}