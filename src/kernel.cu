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
size_t array_index(size_t row, size_t col, array_info *info){
    // helper function to find the array index
    return (row * info->cols) + col;
}

__device__
Position indexDiff(size_t row, size_t col, RadarData_t *radarData, size_t radarPointIdx, 
                   array_info *radarInfo, array_info *mapInfo, array_rel *mapRel){
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
Position index_to_position(size_t row, size_t col, array_info *info, array_rel *relation){
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
                      RadarData_t *radarData, 
                      array_info *mapInfo, 
                      array_rel* mapRel, 
                      array_info* radarInfo,
                      distInfo_t* distributionInfo,
                      radarId_t *radarIds)
    {
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
// Camera point stuff

__global__ 
void camPointKernel(mapType_t* gaussMap, 
                    float *camData, 
                    array_info *mapInfo, 
                    array_rel* mapRel, 
                    array_info* camInfo,
                    distInfo_t* distributionInfo,
                    camVal_t *camClassVals,
                    array_info* camClassInfo){
    /*
    For every camera point, go through each cell in the map and determine the PDF
    value iff the cell is within the cutoff radius. If it is, add the value of 
    the PDF to the gaussMap. This function also records the point's PDF value and
    class in the camClassVals array. This includes the pdf and the class with the
    pdf as the most significant 32 bits of the 64 bit element size array. We use 
    a union to get this value as an unsigned long long int for an atomicMax that
    will store the maximum of either the value in the array or the value passed
    as input (what is calculated for this particular camera point). This allows us
    to keep a map of classes based on camera points, keeping only the class data 
    originating from the closest camera point if overlap occurs. 
    */

    // class vals in the camera list are 1 indexed (zero is background)
    // the config list is provided as a zero-indexed list, so decrement
    uint8_t classVal = __half2ushort_rn(camData[array_index(threadIdx.x, 2, camInfo)] - 1);
    
    for(size_t col = 0; col < mapInfo->cols; col++){
        // find where the cell is relative to the radar point
        Position diff = indexDiff(blockIdx.x, col, 
                                    camData, threadIdx.x, 
                                    camInfo, mapInfo, mapRel);
        // don't calculate the pdf of this cell if it's too far away
        if(diff.radius > distributionInfo[classVal].distCutoff)
            continue;

        float pdfVal = calcPdf(distributionInfo[classVal].stdDev, distributionInfo[classVal].mean, diff.radius);
        atomicAdd(&gaussMap[array_index(blockIdx.x,col,mapInfo)], pdfVal);

        union {
            camVal_t camVal;
            unsigned long long int ulong;
        } cat;

        cat.ulong = 0; // initialize to 0
        cat.camVal.probability = pdfVal;
        cat.camVal.classVal = (uint32_t)camData[array_index(threadIdx.x, 2, camInfo)];
        
        atomicMax((unsigned long long*)&camClassVals[array_index(blockIdx.x,col,camClassInfo)], cat.ulong);
    }
}

void GaussMap::calcCameraMap(){
    checkCudaError(cudaMemcpy(cameraInfo_cuda, &cameraInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));

    // allocate the camera class array
    
    camPointKernel<<<mapInfo.rows,cameraInfo.rows>>>(
        array,
        cameraData,
        mapInfo_cuda,
        mapRel_cuda,
        cameraInfo_cuda,
        cameraDistri_c,
        cameraClassData,
        camClassInfo_cuda
    );

    // wait untill all threads sync
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "camPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

//-----------------------------------------------------------------------------
// maxima locating

__global__
void calcMaxKernel(maxVal_t *isMax, 
                  float* array, array_info *mapInfo,
                  camVal_t *camClassData, array_info *classInfo,
                  uint8_t* windowSizes, radarId_t *radarIds,
                  unsigned int *numMax){
    int row = threadIdx.x;
    int col = blockIdx.x;
    if(row == 0 || row == mapInfo->rows) return;
    if(col == 0 || col == mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    camVal_t camVal = camClassData[array_index(row,col,classInfo)];
    // int windowSize = windowSizes[camVal.classVal];
    int windowSize = 3;
    
    maxVal_t toInsert;
    size_t iterator = 0;
    for(int i = -1 * windowSize; i <= windowSize; i++){
        for(int j = -1 * windowSize; j <= windowSize; j++){
            if(array[array_index(row+i, col+j, mapInfo)] > curVal)
                return;
            if(row+i >=0 && col +j >= 0){
                toInsert.radars[iterator++] = radarIds[array_index(row+i, col+j, mapInfo)].radarId;
            }
        }
    }

    toInsert.isMax = 1;
    toInsert.classVal = (uint8_t) camVal.classVal;
    isMax[array_index(row,col,mapInfo)] = toInsert;
    atomicInc(numMax, 0xffffffff);  // max of unsigned int
}

std::pair<array_info,float*> GaussMap::calcMax(){
    maxVal_t *isMax_cuda;
    checkCudaError(cudaMalloc(&isMax_cuda, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols));

    // initialize isMax to 0
    checkCudaError(cudaMemset(isMax_cuda, 0, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols));

    dim3 maxGridSize(mapInfo.rows, 1, 1);
    dim3 maxBlockSize(mapInfo.cols, 1, 1);

    unsigned int *numMax;
    checkCudaError(cudaMalloc(&numMax, sizeof(unsigned int)));
    checkCudaError(cudaMemset(numMax, 0, sizeof(unsigned int)));

    calcMaxKernel<<<maxGridSize, maxBlockSize>>>(
        isMax_cuda,
        array,
        mapInfo_cuda,
        cameraClassData,
        camClassInfo_cuda,
        windowSizes,
        radarIds,
        numMax
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calcDerivativeKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
   
    float* ret;
    float* ret_c;

    unsigned int *numMaxima;
    numMaxima = (unsigned int*)malloc(sizeof(unsigned int));
    checkCudaError(cudaMemcpy(numMaxima, numMax, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    safeCudaFree(numMax);

    
    array_info maxInfo;
    maxInfo.rows = *numMaxima;
    maxInfo.cols = 4;
    maxInfo.elementSize = sizeof(float);
    checkCudaError(cudaMalloc(&ret_c, sizeof(float) * maxInfo.rows * maxInfo.cols));
    
    array_info *maxInfo_c;
    checkCudaError(cudaMalloc(&maxInfo_c, sizeof(array_info)));
    checkCudaError(cudaMemcpy(maxInfo_c, &maxInfo, sizeof(array_info), cudaMemcpyHostToDevice));
    
    aggregateMax<<<1,*numMaxima>>>(
        array,
        mapInfo_cuda,
        mapRel_cuda,
        isMax_cuda,
        ret_c,
        radarIds,
        maxInfo_c,
        minCutoff
    );
    
    cudaDeviceSynchronize();
    cudaError_t error2 = cudaGetLastError();
    if(error2 != cudaSuccess){
        std::stringstream ss;
        ss << "aggregateMaxKernel launch failed\n";
        ss << cudaGetErrorString(error2);
        throw std::runtime_error(ss.str());
    }
    
    ret = (float*)malloc(*numMaxima * maxInfo.cols * sizeof(float));
    checkCudaError(cudaMemcpy(ret, ret_c, maxInfo.elementSize * maxInfo.rows * maxInfo.cols, cudaMemcpyDeviceToHost));

    safeCudaFree(ret_c);
    safeCudaFree(isMax_cuda);
    safeCudaFree(maxInfo_c);
    free(numMaxima);
    return std::pair<array_info,float*>(maxInfo,ret);
}

__global__ void aggregateMax(mapType_t *array, array_info *mapInfo, array_rel *mapRel,
                             maxVal_t *isMax, float* ret, radarId_t *radarIds,
                             array_info* maxInfo, float minCutoff){
    size_t maxFound = 0;
    maxVal_t tmp;
    for(size_t row = 0; row < mapInfo->rows; row++){
        for(size_t col = 0; col < mapInfo->cols; col++){
            tmp = isMax[(size_t)(row * mapInfo->cols + col)];
            if(tmp.isMax == 1 && array[row * mapInfo->cols + col] >= minCutoff){
                if(maxFound++ == threadIdx.x){
                    ret[array_index(threadIdx.x, 0, maxInfo)] = ((float)(row - mapInfo->rows/2.0) * -1.0) / mapRel->res;
                    ret[array_index(threadIdx.x, 1, maxInfo)] = (col - mapInfo->cols/2.0) / mapRel->res;
                    ret[array_index(threadIdx.x, 2, maxInfo)] = tmp.classVal;
                    ret[array_index(threadIdx.x, 3, maxInfo)] = array[row * mapInfo->cols + col];
                }
            }
        }
    }
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