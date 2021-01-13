/*****************************************************************************
 * This contains the implementation of many different classes and the device *
 * functions used in various kernels. Keeping the implementation in one file *
 * allows the linker to perform link time optimization. Although they can be *
 * seperated, it yeilds a major drawback on performance at runtime.          *
 ****************************************************************************/
#include <sstream>
#include <math_constants.h>     // CUDART_PI_F
#include "gaussMap.cuh"
#include <iostream>

__device__ __forceinline__
size_t array_index(size_t row, size_t col, const array_info *info){
    // helper function to find the array index
    return (row * info->cols) + col;
}

__device__
Position indexDiff(size_t row, size_t col, const RadarData_t *radarData, size_t radarPointIdx, 
                   const array_info *radarInfo, const array_info *mapInfo, const array_rel *mapRel){
    // Calculate the position of the cell at (row,col) relative to the radar point at 
    // radarPointIdx
    Position pos = index_to_position(row, col, mapInfo, mapRel);
    
    float rPosx = radarData[array_index(radarPointIdx, 0, radarInfo)];
    float rPosy = radarData[array_index(radarPointIdx, 1, radarInfo)];
    // printf("rpos %d x: %f, y: %f\n", threadIdx.x, rPosx, rPosy);

    Position difference(
        pos.x - rPosx,
        pos.y - rPosy
    );
    return difference;
}

__device__ 
Position index_to_position(size_t row, size_t col, const array_info *info, const array_rel *relation){
    // find the position from center of map given cell index
    float center_x = (float)(info->cols/2.0);
    float center_y = (float)(info->rows/2.0);
    float x_offset = (col - center_x);
    float y_offset = (row - center_y) * -1;     // flip the y axis so + is in the direction of travel

    Position ret(
        x_offset / (float)relation->res,
        y_offset / (float)relation->res
    );
    return ret;
}

__device__ __forceinline__
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

    for(size_t col = 0; col < mapInfo->cols; col++){
        // find where the cell is relative to the radar point
        Position diff = indexDiff(blockIdx.x, col, 
                                    radarData, threadIdx.x, 
                                    radarInfo, mapInfo, mapRel);
        // don't calculate the pdf of this cell if it's too far away
        if(diff.radius > distributionInfo->distCutoff)
            continue;

        float pdfVal = calcPdf(distributionInfo->stdDev, distributionInfo->mean, diff.radius);
        // printf("pdf: %f\n", pdfVal);
        atomicAdd(&gaussMap[array_index(blockIdx.x,col,mapInfo)], pdfVal);

        un.radData.radarId = threadIdx.x;
        un.radData.probability = pdfVal;
        atomicMax((unsigned long long int*)&radarIds[array_index(blockIdx.x, col, mapInfo)], un.ulong);

    }
}

void GaussMap::calcRadarMap(){

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    checkCudaError(cudaMemcpy(radarInfo_cuda, &radarInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));

    // dispatch the kernel with `numPoints x mapInfo.rows` threads
    radarPointKernel<<<mapInfo.rows,radarInfo.rows>>>(
        array,
        radarData,
        mapInfo_cuda,
        mapRel_cuda,
        radarInfo_cuda,
        radarDistri_c,
        radarIds
    );
    
    // wait untill all threads sync
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "radarPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

}

//-----------------------------------------------------------------------------
// maxima locating

__global__
void calcMaxKernel(maxVal_t *isMax, 
                  const float* array, 
                  const array_info *mapInfo,
                  const radarId_t *radarIds){
    int row = threadIdx.x;
    int col = blockIdx.x;
    if(row == 0 || row == mapInfo->rows) return;
    if(col == 0 || col == mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    maxVal_t *toInsert;
    toInsert = &isMax[array_index(row,col,mapInfo)];
    size_t iterator = 0;
    for(int i = -3; i <= 3; i++){
        for(int j = -3; j <= 3; j++){
            if(array[array_index(row+i, col+j, mapInfo)] > curVal)
                return;
            if(row+i >= 0 && col+j >= 0)
                toInsert->radars[iterator++] = radarIds[array_index(row+i, col+j, mapInfo)].radarId;
        }
    }

    toInsert->isMax = 1;
    toInsert->classVal = 0;
}

__device__ __forceinline__
float calcMean(size_t col, 
               const int16_t* radars, 
               const RadarData_t *radarData, 
               const array_info *radarInfo)
{
    float total = 0;
    size_t numPoints = 0;
    for(size_t i = 0; i < 49; i++){
        if(radars[i] == -1) continue;

        total += radarData[array_index(radars[i], col, radarInfo)];
        numPoints++;
    }
    
    return (total/numPoints);
}

__global__ 
void aggregateMax(const mapType_t *array, 
                  const array_info *mapInfo, 
                  const array_rel *mapRel,
                  const maxVal_t *isMax, 
                  float* ret, 
                  const radarId_t *radarIds,
                  const array_info* maxInfo, 
                  float minCutoff,
                  const RadarData_t *radarData, 
                  const array_info *radarInfo)
{
    // creates an array with the return information in the form of:
    // [row, col, class, pdfVal, vx, vy]
    size_t maxFound = 0;
    maxVal_t tmp;
    for(size_t row = 0; row < mapInfo->rows; row++){
        for(size_t col = 0; col < mapInfo->cols; col++){
            tmp = isMax[(size_t)(row * mapInfo->cols + col)];
            if(tmp.isMax == 1 && array[row * mapInfo->cols + col] >= minCutoff){
                if(maxFound++ == threadIdx.x){
                    ret[array_index(threadIdx.x, 0, maxInfo)] = ((float)(row - mapInfo->rows/2.0) * -1.0) / mapRel->res;
                    ret[array_index(threadIdx.x, 1, maxInfo)] = (col - mapInfo->cols/2.0) / mapRel->res;
                    ret[array_index(threadIdx.x, 2, maxInfo)] = radarData[array_index(tmp.radars[49/2],3, radarInfo)];
                    ret[array_index(threadIdx.x, 3, maxInfo)] = array[row * mapInfo->cols + col];
                    ret[array_index(threadIdx.x, 4, maxInfo)] = calcMean(8, tmp.radars, radarData, radarInfo);
                    ret[array_index(threadIdx.x, 5, maxInfo)] = calcMean(9, tmp.radars, radarData, radarInfo);
                }
            }
        }
    }
}

std::pair<array_info,float*> GaussMap::calcMax(){
    maxVal_t *isMax_cuda;
    checkCudaError(cudaMalloc(&isMax_cuda, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols));

    // initialize isMax to 0
    checkCudaError(cudaMemset(isMax_cuda, 0, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols));

    dim3 maxGridSize(mapInfo.rows, 1, 1);
    dim3 maxBlockSize(mapInfo.cols, 1, 1);

    calcMaxKernel<<<maxGridSize, maxBlockSize>>>(
        isMax_cuda,
        array,
        mapInfo_cuda,
        radarIds
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calcDerivativeKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

    // copy back to host so we can iterate over it
    maxVal_t *isMax = (maxVal_t*)calloc(sizeof(maxVal_t), mapInfo.rows * mapInfo.cols);
    checkCudaError(cudaMemcpy(isMax, isMax_cuda, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols, cudaMemcpyDeviceToHost));
    
    float *arrayTmp = (float*)calloc(sizeof(float), mapInfo.rows * mapInfo.cols);
    checkCudaError(cudaMemcpy(arrayTmp, array, sizeof(float) * mapInfo.rows * mapInfo.cols, cudaMemcpyDeviceToHost));

    // find the number of maxima
    // this can be optimized later
    size_t numMax = 0;
    maxVal_t tmp;
    for(size_t row = 0; row < mapInfo.rows; row++){
        for(size_t col = 0; col < mapInfo.cols; col++){
            tmp = isMax[(size_t)(row * mapInfo.cols + col)];
            if(tmp.isMax == 1 && arrayTmp[row * mapInfo.cols + col] >= minCutoff){
                numMax++;
            }
        }
    }
    
    array_info maxData;
    maxData.cols = 6;
    maxData.rows = numMax;
    maxData.elementSize = sizeof(float);

   
    array_info *maxData_c;
    checkCudaError(cudaMalloc(&maxData_c, sizeof(array_info)));
    checkCudaError(cudaMemcpy(maxData_c, &maxData, sizeof(array_info), cudaMemcpyHostToDevice));

    float *ret, *ret_c;
    checkCudaError(cudaMalloc(&ret_c, sizeof(float) * maxData.rows * maxData.cols));

    aggregateMax<<<1, numMax>>>(
        array,
        mapInfo_cuda,
        mapRel_cuda,
        isMax_cuda,
        ret_c,
        radarIds,
        maxData_c,
        minCutoff,
        radarData,
        radarInfo_cuda
    );

    cudaDeviceSynchronize();
    cudaError_t error2 = cudaGetLastError();
    if(error2 != cudaSuccess){
        std::stringstream ss;
        ss << "aggregateMaxKernel launch failed\n";
        ss << cudaGetErrorString(error2);
        throw std::runtime_error(ss.str());
    }

    ret = (float*)malloc(maxData.size());
    checkCudaError(cudaMemcpy(ret, ret_c, maxData.size(), cudaMemcpyDeviceToHost));

    safeCudaFree(ret_c);
    safeCudaFree(isMax_cuda);
    safeCudaFree(maxData_c);

    return std::pair<array_info,float*>(maxData,ret);
}

//-----------------------------------------------------------------------------
// Association kernel

__global__
void associateCameraKernel(
    const RadarData_t *radarData,
    const array_info *radarInfo,
    const float* camData,
    const array_info *camInfo,
    float* results,
    const array_info *resultInfo
){
    /*
    radarData: [row, col, class, pdfVal, vx, vy]
    cameraData: [x,y,class]
    */
    extern __shared__ float spaceMap[];
    array_info spaceMapInfo;
    
    spaceMapInfo.rows = radarInfo->rows;
    spaceMapInfo.cols = camInfo->rows;
    spaceMapInfo.elementSize = sizeof(float);

    int row = threadIdx.x;
    int col = threadIdx.y;

    float camX, camY;
    float radX, radY;
    camX = camData[array_index(col, 0, camInfo)];
    camY = camData[array_index(col, 1, camInfo)];

    radX = radarData[array_index(row, 0, radarInfo)];
    radY = radarData[array_index(row, 1, radarInfo)];

    float distance = hypotf(camX-radX, camY-radY);
    spaceMap[array_index(row,col,&spaceMapInfo)] = distance;
    printf("%f\n", distance);
    __syncthreads();
}

std::pair<array_info,float*> GaussMap::associateCamera(){
    // calculate the radar's maxima
    std::pair<array_info,float*> maxima = calcMax();
    array_info *maximaInfo;
    checkCudaError(cudaMalloc(&maximaInfo, sizeof(array_info)));
    checkCudaError(cudaMemcpy(maximaInfo, &(maxima.first), cudaMemcpyHostToDevice));

    array_info assocInfo, *assocInfo_c;
    assocInfo.rows = maxima.first.rows + camInfo.rows;
    assocInfo.cols = 6; // [x,y,vx,vy,class,isValid]
    assocInfo.elementSize = sizeof(float);
    
    float* associated;
    checkCudaError(cudaMalloc(&associated, assocInfo.size()));
    checkCudaError(cudaMemset(associated, 0, assocInfo.size()));
    checkCudaError(cudaMalloc(&assocInfo_c, sizeof(array_info)));
    checkCudaError(cudaMemcpy(assocInfo_c, &assocInfo, sizeof(array_info), cudaMemcpyHostToDevice));

    dim3 threadInfo(maxima.first.rows, camInfo.rows);

    associateCameraKernel<<<1, threadInfo, radarInfo.rows*camInfo.rows*sizeof(float)>>>(
        maxima.second,
        maximaInfo,
        camData,
        camInfo_cuda,
        associated,
        assocInfo_c
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "associateCameraKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
    
    return std::pair<array_info,float*>(assocInfo,associated);
}

//-----------------------------------------------------------------------------
// Position class implementation
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

//-----------------------------------------------------------------------------
