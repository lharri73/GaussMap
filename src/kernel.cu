/*****************************************************************************
 * This contains the implementation of many different classes and the device *
 * functions used in various kernels. Keeping the implementation in one file *
 * allows the linker to perform link time optimization. Although they can be *
 * seperated, it yeilds a major drawback on performance at runtime.          *
 ****************************************************************************/
#include <sstream>
#include <math_constants.h>     // CUDART_PI_F
#include "gaussMap.cuh"

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
                      distInfo_t* distributionInfo){
                          
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
        radarDistri_c
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
                  uint8_t* windowSizes){
    int row = threadIdx.x;
    int col = blockIdx.x;
    if(row == 0 || row == mapInfo->rows) return;
    if(col == 0 || col == mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    camVal_t camVal = camClassData[array_index(row,col,classInfo)];
    int windowSize = windowSizes[camVal.classVal];

    for(int i = -1 * windowSize; i <= windowSize; i++){
        for(int j = -1 * windowSize; j <= windowSize; j++){
            if(array[array_index(row+i, col+j, mapInfo)] > curVal)
                return;
        }
    }

    maxVal_t toInsert;
    toInsert.isMax = 1;
    toInsert.classVal = (uint8_t) camVal.classVal;
    isMax[array_index(row,col,mapInfo)] = toInsert;
}

std::vector<float> GaussMap::calcMax(){
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
        cameraClassData,
        camClassInfo_cuda,
        windowSizes
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
    


    // now we don't need the device memory since it's on the host
    checkCudaError(cudaFree(isMax_cuda));

    std::pair<float,float> center(mapInfo.cols/2, mapInfo.rows/2);

    maxVal_t tmp;
    std::vector<float> ret;   // stored as (row,col,class,row,col,class,row,col,class,...)
    for(size_t row = 0; row < mapInfo.rows; row++){
        for(size_t col = 0; col < mapInfo.cols; col++){
            tmp = isMax[(size_t)(row * mapInfo.cols + col)];
            if(tmp.isMax == 1 && arrayTmp[row * mapInfo.cols + col] >= minCutoff){
                ret.push_back(((row - center.second) * -1) / mapRel.res);
                ret.push_back((col - center.first) / mapRel.res);
                ret.push_back(tmp.classVal);
                ret.push_back(arrayTmp[row * mapInfo.cols + col]);
            }
        }
    }

    free(arrayTmp);
    free(isMax);
    return ret;
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