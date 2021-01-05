/*****************************************************************************
 * This contains the implementation of many different classes and the device *
 * functions used in various kernels. Keeping the implementation in one file *
 * allows the linker to perform link time optimization. Although they can be *
 * seperated, it yeilds a major drawback on performance at runtime.          *
 ****************************************************************************/
#include <sstream>
#include <math_constants.h>     // CUDART_PI_F
#include "gaussMap.cuh"

__device__
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
    
    float rPosx = radarData[array_index(threadIdx.x, 0, radarInfo)];
    float rPosy = radarData[array_index(threadIdx.x, 1, radarInfo)];
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

    checkCudaError(cudaMalloc(&mapInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&radarInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship)));
    checkCudaError(cudaMalloc(&radarDistri_c, 3*sizeof(float)));
    
    checkCudaError(cudaMemcpy(mapInfo_cuda, &mapInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarInfo_cuda, &radarInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mapRel_cuda, &mapRel, sizeof(struct Array_Relationship), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarDistri_c, radarDistri, 3*sizeof(float), cudaMemcpyHostToDevice));

    // dispatch the kernel with `numPoints` threads
    radarPointKernel<<<1,radarInfo.rows>>>(
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
                    float* distributionInfo,
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
                          
    for(size_t row = 0; row < mapInfo->rows; row++){
        for(size_t col = 0; col < mapInfo->cols; col++){
            // find where the cell is relative to the radar point
            Position diff = indexDiff(row, col, 
                                      camData, threadIdx.x, 
                                      camInfo, mapInfo, mapRel);
            // don't calculate the pdf of this cell if it's too far away
            if(diff.radius > distributionInfo[2])
                continue;

            float pdfVal = calcPdf(distributionInfo[0], distributionInfo[1], diff.radius);
            atomicAdd(&gaussMap[array_index(row,col,mapInfo)], pdfVal);

            union {
                camVal_t camVal;
                unsigned long long int ulong;
            } cat;

            cat.ulong = 0; // initialize to 0
            cat.camVal.probability = pdfVal;
            cat.camVal.classVal = (uint32_t)camData[array_index(threadIdx.x, 2, camInfo)];
            
            atomicMax((unsigned long long*)&camClassVals[array_index(row,col,camClassInfo)], cat.ulong);
        }
    }
}

void GaussMap::calcCameraMap(){
    camClassInfo.rows = mapInfo.rows;
    camClassInfo.cols = mapInfo.cols;
    camClassInfo.elementSize = sizeof(struct CamVal);

    checkCudaError(cudaMalloc(&cameraInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&camClassInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&cameraDistri_c, 3*sizeof(float)));

    checkCudaError(cudaMemcpy(cameraInfo_cuda, &cameraInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(camClassInfo_cuda, &camClassInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(cameraDistri_c, cameraDistri, 3*sizeof(float), cudaMemcpyHostToDevice));

    // allocate the camera class array
    checkCudaError(cudaMalloc(&cameraClassData, camClassInfo.elementSize * camClassInfo.rows * camClassInfo.cols));
    checkCudaError(cudaMemset(cameraClassData, 0, camClassInfo.elementSize * camClassInfo.rows * camClassInfo.cols));

    camPointKernel<<<1,cameraInfo.rows>>>(
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
void calcMaxKernel(uint8_t *isMax, float* array, 
                   array_info *mapInfo){
    int row = threadIdx.x;
    int col = blockIdx.x;
    if(row == 0 || row == mapInfo->rows) return;
    if(col == 0 || col == mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    for(int i = -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            if(array[array_index(row+i, col+j, mapInfo)] > curVal)
                return;
        }
    }

    isMax[array_index(row,col,mapInfo)] = 1;

}

std::vector<uint16_t> GaussMap::calcMax(){
    uint8_t *isMax_cuda;
    checkCudaError(cudaMalloc(&isMax_cuda, sizeof(uint8_t) * mapInfo.rows * mapInfo.cols));

    // initialize isMax to 0
    checkCudaError(cudaMemset(isMax_cuda, 0, sizeof(uint8_t) * mapInfo.rows * mapInfo.cols));

    dim3 maxGridSize(mapInfo.rows, 1, 1);
    dim3 maxBlockSize(mapInfo.cols, 1, 1);

    calcMaxKernel<<<maxGridSize, maxBlockSize>>>(
        isMax_cuda,
        array,
        mapInfo_cuda
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
    uint8_t *isMax = (uint8_t*)calloc(sizeof(uint8_t), mapInfo.rows * mapInfo.cols);
    checkCudaError(cudaMemcpy(isMax, isMax_cuda, sizeof(uint8_t) * mapInfo.rows * mapInfo.cols, cudaMemcpyDeviceToHost));
    
    // now we don't need the device memory since it's on the host
    checkCudaError(cudaFree(isMax_cuda));

    std::vector<uint16_t> ret;   // stored as (row,col,row,col,row,col,...)
    for(uint16_t row = 0; row < mapInfo.rows; row++){
        for(uint16_t col = 0; col < mapInfo.cols; col++){
            if(isMax[(size_t)(row * mapInfo.cols + col)] == 1){
                ret.push_back(row);
                ret.push_back(col);
            }
        }
    }
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
// 