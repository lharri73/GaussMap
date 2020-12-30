#include <sstream>
#include <math_constants.h>     // CUDART_PI_F
#include "gaussMap.cuh"


__device__
size_t array_index(size_t row, size_t col, array_info *info){
    // helper function to find the array index
    return row * info->cols + col;
}

template <typename T>
__device__
float radiusFromPos(T x, T y){
    // return the radius from the position from origin (at center of map)
    // x and y in meters
    return hypotf((float)x, (float)y);
}

__device__
Position indexDiff(size_t row, size_t col, RadarData_t *radarData, size_t radarPointIdx, 
                   array_info *radarInfo, array_info *mapInfo, array_rel *mapRel){
    // Calculate the position of the cell at (row,col) relative to the radar point at 
    // radarPointIdx
    Position pos = index_to_position(row, col, mapInfo, mapRel);
    
    float rPosx = radarData[array_index(threadIdx.x, 0, radarInfo)];
    float rPosy = radarData[array_index(threadIdx.x, 1, radarInfo)];
    printf("rpos %d x: %f, y: %f\n", threadIdx.x, rPosx, rPosy);

    Position difference(
        pos.x - rPosx,
        pos.y - rPosy
    );
    return difference;
}

__device__ 
Position index_to_position(size_t row, size_t col, array_info *info, array_rel *relation){
    // find the position from center of map given cell index
    float center_x = (float)(info->cols/2.0);
    float center_y = (float)(info->rows/2.0);
    float x_offset = col - center_x;
    float y_offset = (row - center_y) * -1;     // flip the y axis so + is in the direction of travel

    Position ret(
        x_offset / (float)relation->res,
        y_offset / (float)relation->res
    );
    return ret;
}

__device__
float calcPdf(float stdDev, float mean, float radius){
    // calculate the pdf of the given radar point based on the radius
    float variance = pow(stdDev, 2);
    float first = (1 / stdDev) * rsqrt(2 * CUDART_PI_F);
    float exponent = -1 * pow(radius - mean, 2) / (2 * variance);
    float second = exp(exponent);
    return first*second;
}

__global__ 
void radarPointKernel(mapType_t* gaussMap, 
                      RadarData_t *radarData, 
                      array_info *mapInfo, 
                      array_rel* mapRel, 
                      array_info* radarInfo,
                      float* distributionInfo){
    for(size_t row = 0; row < mapInfo->rows; row++){
        for(size_t col = 0; col < mapInfo->cols; col++){
            // find where the cell is relative to the radar point
            Position diff = indexDiff(row, col, 
                                              radarData, threadIdx.x, 
                                              radarInfo, mapInfo, mapRel);
            // don't calculate the pdf of this cell if it's too far away
            if(diff.radius > distributionInfo[2])
                continue;

            float pdfVal = calcPdf(distributionInfo[0], distributionInfo[1], diff.radius);
            // printf("pdf: %f\n", pdfVal);
            atomicAdd(&gaussMap[array_index(row,col,mapInfo)], pdfVal);
        }
    }
}

void GaussMap::calcRadarMap(){

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    array_info *tmpa, *tmpb;
    array_rel *tmpc;
    tmpa = (array_info*)malloc(sizeof(struct Array_Info));
    tmpb = (array_info*)malloc(sizeof(struct Array_Info));
    tmpc = (array_rel*)malloc(sizeof(struct Array_Relationship));
    memcpy(tmpa, &mapInfo, sizeof(struct Array_Info));
    memcpy(tmpb, &radarInfo, sizeof(struct Array_Info));
    memcpy(tmpc, &mapRel, sizeof(struct Array_Relationship));

    checkCudaError(cudaMalloc(&mapInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&radarInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship)));
    checkCudaError(cudaMalloc(&radarDistri_c, 3*sizeof(float)));
    checkCudaError(cudaMemcpy(mapInfo_cuda, tmpa, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarInfo_cuda, tmpb, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mapRel_cuda, tmpc, sizeof(struct Array_Relationship), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarDistri_c, radarDistri, 3*sizeof(float), cudaMemcpyHostToDevice));

    free(tmpa);
    free(tmpb);
    free(tmpc);


    // dispatch the kernel with `numPoints` threads
    printf("numPoints: %lu\n", numPoints);
    radarPointKernel<<<1,numPoints>>>(
        array,
        radarData,
        mapInfo_cuda,
        mapRel_cuda,
        radarInfo_cuda,
        radarDistri_c
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "radarPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

    // wait untill all threads sync
    cudaDeviceSynchronize();
}


//-----------------------------------------------------------------------------
__device__ 
Position::Position(float X, float Y) : x(X), y(Y){
    radius = hypotf(x,y);
}

__device__
Position::Position(){

}

__device__
void Position::recalc(){
    radius = hypotf(x,y);
}