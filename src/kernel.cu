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
    return row * info->cols + col;
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
    radarPointKernel<<<1,numPoints>>>(
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
// Derivative code implementation
__global__
void calcDerivativeKernel(float* f, array_info *fInfo, float* fPrime, array_info *fPrimeInfo){
    // https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences 
    // ^ 5th equation
    size_t row,col;
    row = threadIdx.x +1;
    col = blockIdx.x +1;
    float first, second, third, fourth;
    first = f[array_index(row+1, col+1, fInfo)];
    second = f[array_index(row+1, col-1, fInfo)];
    third = f[array_index(row-1, col+1, fInfo)];
    fourth = f[array_index(row-1, col-1, fInfo)];

    fPrime[array_index(threadIdx.x, blockIdx.x, fPrimeInfo)] = (first - second - third + fourth) / 4.0;
    // printf("%d %d %d | %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

void GaussMap::calcDerivative(){
    primeInfo.rows = mapInfo.rows -2;
    primeInfo.cols = mapInfo.cols -2;
    primeInfo.elementSize = sizeof(float);

    checkCudaError(cudaMalloc(&primeInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMemcpy(primeInfo_cuda, &primeInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    if(mapInfo_cuda == nullptr)
        throw std::runtime_error("radar data must be added before the derivative can be calculated.");
    
    checkCudaError(cudaMalloc(&arrayPrime, sizeof(float) * primeInfo.rows * primeInfo.cols));

    // dispatch the kernel with a single thread per cell
    dim3 primeGridSize(primeInfo.rows);
    dim3 primeBlockSize(primeInfo.cols);
    calcDerivativeKernel<<<primeGridSize,primeBlockSize>>>(
        array,
        mapInfo_cuda,
        arrayPrime,
        primeInfo_cuda
    );
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calcDerivativeKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

    // now do it again...
    primePrimeInfo.rows = primeInfo.rows -2;
    primePrimeInfo.cols = primeInfo.cols -2;
    primePrimeInfo.elementSize = sizeof(float);

    checkCudaError(cudaMalloc(&primePrimeInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMemcpy(primePrimeInfo_cuda, &primePrimeInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMalloc(&arrayPrimePrime, sizeof(float) * primePrimeInfo.rows * primePrimeInfo.cols));

    dim3 primePrimeGridSize(primePrimeInfo.rows);
    dim3 primePrimeBlockSize(primePrimeInfo.cols);
    calcDerivativeKernel<<<primePrimeGridSize,primePrimeBlockSize>>>(
        arrayPrime,
        primeInfo_cuda,
        arrayPrimePrime,
        primePrimeInfo_cuda
    );
    
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calcDerivativeKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

}

//-----------------------------------------------------------------------------
// maxima locating

__global__
void calcMaxKernel(uint8_t *isMax, float* arrayPrime, 
                   array_info *primeInfo, float* arrayPrimePrime, 
                   array_info *primePrimeInfo){

}

std::vector<uint16_t> GaussMap::calcMax(){
    if(!arrayPrime || !arrayPrimePrime) // make sure these are allocated
        throw std::runtime_error("Derivative must be calculated before maxima can be located");

    uint8_t *isMax_cuda;
    checkCudaError(cudaMalloc(&isMax_cuda, sizeof(uint8_t) * primePrimeInfo.rows * primePrimeInfo.cols));

    dim3 maxGridSize(primePrimeInfo.rows);
    dim3 maxBlockSize(primePrimeInfo.cols);

    calcMaxKernel<<<maxGridSize, maxBlockSize>>>(
        isMax_cuda,
        arrayPrime,
        primeInfo_cuda,
        arrayPrimePrime,
        primePrimeInfo_cuda
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
    uint8_t *isMax = (uint8_t*)calloc(sizeof(uint8_t), primePrimeInfo.rows * primePrimeInfo.cols);
    checkCudaError(cudaMemcpy(isMax, isMax_cuda, sizeof(uint8_t) * primePrimeInfo.rows * primePrimeInfo.cols, cudaMemcpyDeviceToHost));
    
    // now we don't need the device memory since it's on the host
    checkCudaError(cudaFree(isMax_cuda));

    std::vector<uint16_t> ret;   // stored as (row,col,row,col,row,col,...)
    for(uint16_t row = 0; row < primePrimeInfo.rows; row++){
        for(uint16_t col = 0; col < primePrimeInfo.cols; col++){
            if(isMax[(size_t)(row * primePrimeInfo.cols + col)]){
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
