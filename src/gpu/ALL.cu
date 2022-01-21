#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif

class valInt_t{
public:
    __device__ valInt_t(float value, unsigned int index): value_(value), index_(index){};
    float value_;
    unsigned int index_;
    __device__ friend bool operator<(const valInt_t &a, const valInt_t &b);
    __device__ friend bool operator<=(const valInt_t &a, const valInt_t &b);
    __device__ friend bool operator>(const valInt_t &a, const valInt_t &b);
    __device__ friend bool operator==(const valInt_t &a, const valInt_t &b);
    __device__ friend bool operator!=(const valInt_t &a, const valInt_t &b);

};
__device__ bool operator<(const valInt_t &a, const valInt_t &b){
    return a.value_ < b.value_;
}
__device__ bool operator<=(const valInt_t &a, const valInt_t &b){
    return a.value_ <= b.value_;
}
__device__ bool operator>(const valInt_t &a, const valInt_t &b){
    return a.value_ > b.value_;
}
__device__ bool operator==(const valInt_t &a, const valInt_t &b){
    return a.value_ == b.value_;
}
__device__ bool operator!=(const valInt_t &a, const valInt_t &b){
    return a.value_ != b.value_;
}

__device__ void merge(valInt_t arr[], int l, int m, int r);
__device__ void mergeSort(valInt_t arr[], int n);

__global__
void setSpaceMap(
    const RadarData_t* __restrict__ radarData,
    const array_info* __restrict__ radarInfo,
    const float* __restrict__ camData,
    const array_info* __restrict__ camInfo,
    float* __restrict__ spaceMap,
    const array_info* __restrict__ spaceMapInfo)
{ 
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    float camX, camY;
    float radX, radY;
    camX = camData[array_index(col, 0, camInfo)];
    camY = camData[array_index(col, 1, camInfo)];
    
    radX = radarData[array_index(row, 0, radarInfo)];
    radY = radarData[array_index(row, 1, radarInfo)];
    
    // calculate the pairwise distance for each camera,radar point
    float distance = hypotf(camX-radX, camY-radY);
    spaceMap[array_index(row,col,spaceMapInfo)] = distance;
}

__global__
void associateCameraKernel(
    const RadarData_t* __restrict__ radarData,
    const array_info* __restrict__ radarInfo,
    const float* __restrict__ camData,
    const array_info* __restrict__ camInfo,
    float* __restrict__ results,
    const array_info* __restrict__ resultInfo,
    float* __restrict__ spaceMap,
    const array_info * __restrict__ spaceMapInfo,
    float adjustFactor
){
    /* spaceMap: 
        rows: radarData
        cols: camData
        values: euc. distance between each 
      radarData: [row, col, class, pdfVal, vx, vy]
      cameraData: [x,y,class]
      ret: [x,y,vx,vy,class,isValid]
    */
   
    // we only need one because we need to find the max of the threadIdx.xumn vector
    
    float min = 3e38;// This is near the max value of a float //spaceMap[array_index(0, threadIdx.x, &spaceMapInfo)];
    // float cur;
    int minIndex = -1;

    valInt_t *values = (valInt_t*)malloc(spaceMapInfo->rows * sizeof(valInt_t));
    for(size_t i = 0; i < spaceMapInfo->rows; i++)
        values[i] = valInt_t(spaceMap[array_index(i,threadIdx.x,spaceMapInfo)], i);
    
    mergeSort(values, spaceMapInfo->rows);

    // the association distance is decided based on the distance from the vehicle (from 0)
    // the idea is the farther from the vehicle, the less acurate the detection will be
    #ifdef NUSCENES
    float cutoff = hypotf(camData[array_index(threadIdx.x, 0, camInfo)], camData[array_index(threadIdx.x, 1, camInfo)]) * adjustFactor;
    #else
    float cutoff = camData[array_index(threadIdx.x, 0, camInfo)] * adjustFactor;
    #endif
    float curWidth = camData[array_index(threadIdx.x, 3,camInfo)];
    
    int curIndex;
    int i = 0;
    float widthDiff = 1e4;
    while(widthDiff > curWidth && i < spaceMapInfo->rows){  // iterate through all radar detections in sorted order
        curIndex = values[i].index_;
        min = values[i].value_;
        widthDiff = fabsf(radarData[array_index(curIndex,1,radarInfo)] - camData[array_index(threadIdx.x,1,camInfo)]);
        i++;
    }
    if(i != spaceMapInfo->rows)
        minIndex = curIndex;    // found a good radar detection

    if(minIndex > -1 && min <= cutoff){
        spaceMap[array_index(minIndex, threadIdx.x, spaceMapInfo)] = -1.0; // a signal to join these two points
    }else{
        int resultRow = radarInfo->rows+threadIdx.x;
        results[array_index(resultRow, 0, resultInfo)] = camData[array_index(threadIdx.x, 0, camInfo)];     //x
        results[array_index(resultRow, 1, resultInfo)] = camData[array_index(threadIdx.x, 1, camInfo)];     //y
        results[array_index(resultRow, 2, resultInfo)] = 0;                                         //vx
        results[array_index(resultRow, 3, resultInfo)] = 0;                                         //vy
        results[array_index(resultRow, 4, resultInfo)] = camData[array_index(threadIdx.x, 2, camInfo)];     //class
        results[array_index(resultRow, 5, resultInfo)] = 1;                                         //isValid
    }
    free(values);
}

__global__
void joinFeatures(
    const RadarData_t* __restrict__ radarData,
    const array_info* __restrict__ radarInfo,
    const float* __restrict__ camData,
    const array_info * __restrict__ camInfo,
    float* __restrict__ results,
    const array_info* __restrict__ resultInfo,
    float* __restrict__ spaceMap,
    const array_info* __restrict__ spaceMapInfo
){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if(spaceMap[array_index(row, col, spaceMapInfo)] < 0.0){
        // received the signal to fuse the objects
        results[array_index(blockIdx.x, 0, resultInfo)] = radarData[array_index(row, 0, radarInfo)]; //x
        results[array_index(blockIdx.x, 1, resultInfo)] = radarData[array_index(row, 1, radarInfo)]; //y
        results[array_index(blockIdx.x, 2, resultInfo)] = radarData[array_index(row, 2, radarInfo)]; //vx
        results[array_index(blockIdx.x, 3, resultInfo)] = radarData[array_index(row, 3, radarInfo)]; //vy
        results[array_index(blockIdx.x, 4, resultInfo)] = camData[array_index(col, 2, camInfo)];     //class
        results[array_index(blockIdx.x, 5, resultInfo)] = 1;                                         //isValid
        return;
    }

    if(threadIdx.x == 0 && results[array_index(blockIdx.x, 5, resultInfo)] != 1.0){
        // this is just a radar detection
        results[array_index(blockIdx.x, 0, resultInfo)] = radarData[array_index(row, 0, radarInfo)]; //x
        results[array_index(blockIdx.x, 1, resultInfo)] = radarData[array_index(row, 1, radarInfo)]; //y
        results[array_index(blockIdx.x, 2, resultInfo)] = radarData[array_index(row, 2, radarInfo)]; //vx
        results[array_index(blockIdx.x, 3, resultInfo)] = radarData[array_index(row, 3, radarInfo)]; //vy
        results[array_index(blockIdx.x, 4, resultInfo)] = 0;                                         //class
        results[array_index(blockIdx.x, 5, resultInfo)] = 1;                                         //isValid
    }
}


// Simple kernel that transforms the camera/radar input array to the result
// format. This is used when there are no camera or radar points for the
// association kernel to associate. 
__global__
void singleElementResult(const float* __restrict__ radarData,
                         const array_info* __restrict__ radarInfo,
                         const float* __restrict__ camData,
                         const array_info* __restrict__ camInfo,
                         float* __restrict__ results,
                         const array_info *resultInfo)
{
    if(camInfo->rows == 0){
        results[array_index(blockIdx.x, 0, resultInfo)] = radarData[array_index(blockIdx.x, 0, radarInfo)];
        results[array_index(blockIdx.x, 1, resultInfo)] = radarData[array_index(blockIdx.x, 1, radarInfo)];
        results[array_index(blockIdx.x, 2, resultInfo)] = radarData[array_index(blockIdx.x, 3, radarInfo)];
        results[array_index(blockIdx.x, 3, resultInfo)] = radarData[array_index(blockIdx.x, 4, radarInfo)];
        results[array_index(blockIdx.x, 4, resultInfo)] = 1.0;
    }else{
        results[array_index(blockIdx.x, 0, resultInfo)] = camData[array_index(blockIdx.x, 0, camInfo)];
        results[array_index(blockIdx.x, 1, resultInfo)] = camData[array_index(blockIdx.x, 1, camInfo)];
        results[array_index(blockIdx.x, 2, resultInfo)] = 0.0;
        results[array_index(blockIdx.x, 3, resultInfo)] = 0.0;
        results[array_index(blockIdx.x, 4, resultInfo)] = camData[array_index(blockIdx.x, 2, camInfo)];
    }

}

//-----------------------------------------------------------------------------
// mergeSort
// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__device__
void merge(valInt_t* arr, int l, int m, int r);
 
/* Iterative mergesort function to sort arr[0...n-1] */
__device__
void mergeSort(valInt_t* arr, int n)
{
   int curr_size;  // For current size of subarrays to be merged
                   // curr_size varies from 1 to n/2
   int left_start; // For picking starting index of left subarray
                   // to be merged
 
   // Merge subarrays in bottom up manner.  First merge subarrays of
   // size 1 to create sorted subarrays of size 2, then merge subarrays
   // of size 2 to create sorted subarrays of size 4, and so on.
   for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
   {
       // Pick starting point of different subarrays of current size
       for (left_start=0; left_start<n-1; left_start += 2*curr_size)
       {
           // Find ending point of left subarray. mid+1 is starting
           // point of right
           int mid = min(left_start + curr_size - 1, n-1);
 
           int right_end = min(left_start + 2*curr_size - 1, n-1);
 
           // Merge Subarrays arr[left_start...mid] & arr[mid+1...right_end]
           merge(arr, left_start, mid, right_end);
       }
   }
}
 
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__device__
void merge(valInt_t arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    valInt_t *L = (valInt_t*)malloc(n1 * sizeof(valInt_t));
    valInt_t *R = (valInt_t*)malloc(n2 * sizeof(valInt_t));
 
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}
#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif

__global__
void calcMaxKernel(maxVal_t* __restrict__ isMax, 
                  const float* __restrict__ array, 
                  const array_info* __restrict__ mapInfo,
                  const radarId_t* __restrict__ radarIds,
                  const array_info* __restrict__ idInfo,
                  unsigned short windowSize,
                  int16_t* __restrict__ windowIds,
                  const array_info* __restrict__ windowIdInfo){
    /*
    for every point in the map, check a 7x7 grid (up to the edges) to see if it
    is a local max. if it is, put the ids of each radar in this range into a list
    so the algorithm can use it later
    */
    int col = threadIdx.x;
    int row = blockIdx.x;
    size_t iterator = 0;

    if(row == 0 || row >= mapInfo->rows) return;
    if(col == 0 || col >= mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    
    maxVal_t *toInsert;
    toInsert = &isMax[array_index(row,col,mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    for(int i = -windowSize; i <= windowSize; i++){
        for(int j = -windowSize; j <= windowSize; j++){
            if(row+i > mapInfo->rows || col+j > mapInfo->cols) {
                windowIds[array_index(array_index(row,col,mapInfo),iterator++,windowIdInfo)] = -1;
                continue;
            }
            
            if(array[array_index(row+i, col+j, mapInfo)] > curVal){
                toInsert->isMax = 0;
                return;
            }
            
            if(row+i >= 0 && col+j >= 0){
                int16_t id = radarIds[array_index(row+i, col+j, mapInfo)].radarId;
                if(id > (int)idInfo->rows || id < -1)
                    id = -1;
                windowIds[array_index(array_index(row,col,mapInfo),iterator++,windowIdInfo)] = id;
            }
        }
    }

    
    toInsert->isMax = 1;
    toInsert->classVal = 0;
    return;
}

__global__ 
void aggregateMax(const mapType_t* __restrict__ array, 
                  const array_info* __restrict__ mapInfo, 
                  const array_rel* __restrict__ mapRel,
                  const maxVal_t* __restrict__ isMax,
                  float* __restrict__ ret, 
                  const array_info* __restrict__ maxInfo,
                  const RadarData_t* __restrict__ radarData, 
                  const array_info* __restrict__ radarInfo, 
                  const int* __restrict__ maximaLocs,
                  const array_info* __restrict__ locsInfo,
                  const int16_t* __restrict__ windowIds,
                  const array_info* __restrict__ windowIdInfo)
{
    // creates an array with the return information in the form of:
    // [row, col, pdfVal, vx, vy, targetId]
    int row,col;
    row = maximaLocs[array_index(threadIdx.x, 0, locsInfo)];
    col = maximaLocs[array_index(threadIdx.x, 1, locsInfo)];
    
    maxVal_t tmp = isMax[array_index(row,col,mapInfo)];
    

    ret[array_index(threadIdx.x, 0, maxInfo)] = (col - mapInfo->cols/2.0) / mapRel->res;
    ret[array_index(threadIdx.x, 1, maxInfo)] = -(row - mapInfo->rows/2.0) / mapRel->res;
    ret[array_index(threadIdx.x, 2, maxInfo)] = array[array_index(row, col, mapInfo)];
    ret[array_index(threadIdx.x, 3, maxInfo)] = calcMean(array_index(row,col,mapInfo), 2, windowIds, windowIdInfo, radarData, radarInfo);
    ret[array_index(threadIdx.x, 4, maxInfo)] = calcMean(array_index(row,col,mapInfo), 3, windowIds, windowIdInfo, radarData, radarInfo);

    // attach the target ID to the maxima.
    // we assume there is only one target for each maxima detection
    int targetId = -1;
    int tmpId;
    int16_t curId;
    for(size_t i = 0; i < searchSize; i++){
        curId = windowIds[array_index(array_index(row,col,mapInfo),i,windowIdInfo)];
        
        if(curId == -1) continue;
        if(curId > radarInfo->rows){
            printf("radarId index out of bounds (aggregateMax) %hd\n", curId);
            break;
        }
        tmpId = radarData[array_index(curId, 5, radarInfo)];
        
        targetId = tmpId == 255 ? -1 : tmpId;
    }
    ret[array_index(threadIdx.x, 5, maxInfo)] = targetId;
}

__device__
float calcMean(size_t cellIndex,
               size_t radarCol, 
               const int16_t* __restrict__ radars, 
               const array_info* __restrict__ idInfo,
               const RadarData_t* __restrict__ radarData, 
               const array_info* __restrict__ radarInfo)
{
    float total = 0.0f;
    float numPoints = 0.0f;
    int16_t curId;
    for(size_t i = 0; i < searchSize; i++){
        curId = radars[array_index(cellIndex,i,idInfo)];
        if(curId == -1) continue;
        CUDA_ASSERT_LT_E(curId, radarInfo->rows, "radarId index out of bounds");
        //if(curId > radarInfo->rows){
        //   printf("radarId index out of bounds (calcMean) %hd\n", curId);
        //    printf("\tindex: %llu. col: %llu. i: %llu\n", cellIndex, radarCol, i);
        //    break;
        //}

        total += radarData[array_index(curId, radarCol, radarInfo)];
        numPoints++;
    }

    // divide by zero is bad. But apparently it's zero!
    if(numPoints == 0)
        return 0.0f;
    
    return (total/numPoints);
}
#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#include "ecocar_fusion/utils.hpp"
#else
#include "gaussMap.cuh"
#include "utils.hpp"
#endif
#include <sstream>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

// allocate this struct in shared memory so we don't have to copy
// it to each kernel when it's needed

// expected to be in: [x,y,vx,vy,wExist,targetId,dxSig,dySig]
void GaussMap::calcRadarMap(){
    if(radarInfo.cols != 8)
        throw std::runtime_error("size of radar data is incorrect. Should be Nx8");

    safeCudaMemcpy2Device(radarInfo_cuda, &radarInfo, sizeof(array_info));

    if(radarInfo.rows == 0){
        printf("no radar points this round\n");
        return;
    }

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
    
    // wait until all threads sync
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "radarPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

// returns the location of the maxima points
// [row, col, pdfVal, vx, vy, targetId]
std::pair<array_info,float*> GaussMap::calcMax(){
    maxVal_t *isMax_cuda;
    cudaError_t error;
    maxVal_t *isMax;
    float *arrayTmp;
    size_t numMax;
    maxVal_t tmp;

    safeCudaMalloc(&isMax_cuda, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols);

    // initialize isMax to 0
    safeCudaMemset(isMax_cuda, 0, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols);

    dim3 maxGridSize(mapInfo.rows, 1, 1);   // blocks per grid
    dim3 maxBlockSize(mapInfo.cols, 1, 1);  // threads per block

    calcMaxKernel<<<maxGridSize, maxBlockSize>>>(
        isMax_cuda,
        array,
        mapInfo_cuda,
        radarIds,
        radarInfo_cuda,
        winSize,
        windowIds,
        windowIdInfo_cuda
    );

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "calMaxKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }

    // copy back to host so we can iterate over it
    isMax = (maxVal_t*)calloc(sizeof(maxVal_t), mapInfo.rows * mapInfo.cols);
    safeCudaMemcpy2Host(isMax, isMax_cuda, sizeof(maxVal_t) * mapInfo.rows * mapInfo.cols);
    
    arrayTmp = (float*)calloc(sizeof(float), mapInfo.rows * mapInfo.cols);
    safeCudaMemcpy2Host(arrayTmp, array, sizeof(float) * mapInfo.rows * mapInfo.cols);

    // find the number of maxima
    // this can be optimized later
    numMax = 0;
    std::vector<int> maximaLocs;     // [row,col,row,col,...]
    for(size_t row = 0; row < mapInfo.rows; row++){
        for(size_t col = 0; col < mapInfo.cols; col++){
            tmp = isMax[(size_t)(row * mapInfo.cols + col)];
            if(tmp.isMax == 1 && arrayTmp[row * mapInfo.cols + col] >= minCutoff){
                numMax++;
                maximaLocs.push_back(row);
                maximaLocs.push_back(col);
            }
        }
    }

    free(isMax);
    free(arrayTmp);

    if(numMax == 0 && radarInfo.rows != 0){
        ROS_ERROR_STREAM("reached invalid maxima configuration. Skipping (bug)");
        array_info early_info;
        early_info.rows = 0;
        early_info.cols = 6;
        early_info.elementSize = sizeof(float);
        return std::make_pair(early_info, nullptr);
    }else if(numMax == 0){
        ROS_WARN_STREAM("No Radar detections this cycle. Cycle time too fast?");
        array_info early_info;
        early_info.rows = 0;
        early_info.cols = 6;
        early_info.elementSize = sizeof(float);
        return std::make_pair(early_info, nullptr);
    }

    // allocate the maxima locations in CUDA
    int *maximaLocs_c;
    safeCudaMalloc(&maximaLocs_c, maximaLocs.size() * sizeof(int));
    safeCudaMemcpy2Device(maximaLocs_c, (int*)maximaLocs.data(), maximaLocs.size() * sizeof(uint32_t));
    array_info maximaLocs_info;
    maximaLocs_info.cols = 2;
    maximaLocs_info.rows = maximaLocs.size() / 2;
    maximaLocs_info.elementSize = sizeof(int);
    
    if(maximaLocs_info.rows != numMax)
        throw std::runtime_error("calcMax(): failed to push all maxima locations to list. Memory leak?");

    array_info *maximaloc_nfo_c;
    safeCudaMalloc(&maximaloc_nfo_c, sizeof(array_info));;
    safeCudaMemcpy2Device(maximaloc_nfo_c, &maximaLocs_info, sizeof(array_info));
    
    array_info maxData;
    maxData.cols = 6;
    maxData.rows = numMax;
    maxData.elementSize = sizeof(float);

    array_info *maxData_c;
    safeCudaMalloc(&maxData_c, sizeof(array_info));
    safeCudaMemcpy2Device(maxData_c, &maxData, sizeof(array_info));

    float *ret_c;
    safeCudaMalloc(&ret_c, maxData.size());
    aggregateMax<<<1, numMax>>>(
        array,
        mapInfo_cuda,
        mapRel_cuda,
        isMax_cuda,
        ret_c,          // max
        maxData_c,      // maxInfo
        radarData,      // radarData
        radarInfo_cuda,
        maximaLocs_c,
        maximaloc_nfo_c,
        windowIds,
        windowIdInfo_cuda
    );

    cudaDeviceSynchronize();
    cudaError_t error2 = cudaGetLastError();
    if(error2 != cudaSuccess){
        std::stringstream ss;
        ss << "aggregateMaxKernel launch failed. Size: 1x" << numMax << '\n';
        ss << cudaGetErrorString(error2);
        throw std::runtime_error(ss.str());
    }

    safeCudaFree(isMax_cuda);
    safeCudaFree(maxData_c);
    safeCudaFree(maximaLocs_c);
    safeCudaFree(maximaloc_nfo_c);

    return std::pair<array_info,float*>(maxData,ret_c);
}

// resets the radar ids back to -1 so we know that 
// they are uninitialized
void GaussMap::setRadarIds(){
    setRadarIdsKernel<<<mapInfo.rows*mapInfo.cols,1>>>(
        radarIds
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "failed to set radar ids to -1\n";
        ss << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

// performs the association between radar and camera detections
// ret: [x,y,vx,vy,class]
std::pair<array_info,float*> GaussMap::associatePair(){

    // calculate the radar's maxima
    std::pair<array_info,float*> maxima = calcMax();    // [row, col, pdfVal, vx, vy, targetId]
    array_info maximaInfo, *maximaInfo_c;
    maximaInfo = maxima.first;

    safeCudaMalloc(&maximaInfo_c, sizeof(array_info));
    safeCudaMemcpy2Device(maximaInfo_c, &maximaInfo, sizeof(array_info));

    array_info assocInfo, *assocInfo_c;
    assocInfo.cols = 6; // [x,y,vx,vy,class,isValid]
    assocInfo.elementSize = sizeof(float);
    
    // unlikely to happen, but prevents a segmentation fault
    if((maximaInfo.rows == 0) != (camInfo.rows == 0)){  // XOR
        // handle the case when there are no radar points ^ no cam points
        assocInfo.rows = maximaInfo.rows + camInfo.rows;
        assocInfo.cols -=1;     // we don't need the valid flag since we know all are valid

        float* associated;
        safeCudaMalloc(&associated, assocInfo.size());
        safeCudaMalloc(&assocInfo_c, sizeof(array_info));
        safeCudaMemcpy2Device(assocInfo_c, &assocInfo, sizeof(array_info));
        
        dim3 blockInfo(maximaInfo.rows + camInfo.rows,1);
        dim3 threadInfo(1,1);

        singleElementResult<<<blockInfo,threadInfo>>>(
            maxima.second,
            maximaInfo_c,
            camData,
            camInfo_cuda,
            associated,
            assocInfo_c
        );

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            std::stringstream ss;
            ss << "singleElementResult kernel launch failed\n";
            ss << cudaGetErrorString(error);
            throw std::runtime_error(ss.str());
        } 

        // move the results back to host memory
        float* ret;
        ret = (float*)malloc(assocInfo.size());
        safeCudaMemcpy2Host(ret, associated, assocInfo.size());
        safeCudaFree(associated);

        return std::pair<array_info,float*> (assocInfo,ret);

    }else if(maximaInfo.rows == 0 && camInfo.rows == 0){
        // handle the case when there are no camera or radar points
        assocInfo.rows = 0;
        assocInfo.cols -=1;
        
        float* earlyRet = (float*)malloc(0);
        return std::pair<array_info,float*>(assocInfo,earlyRet);
    }else{
        // normal situation. there are camera and radar points
        assocInfo.rows = maximaInfo.rows + camInfo.rows;

        float* associated;
        safeCudaMalloc(&associated, assocInfo.size());
        checkCudaError(cudaMemset(associated, 0, assocInfo.size()));
        safeCudaMalloc(&assocInfo_c, sizeof(array_info));
        safeCudaMemcpy2Device(assocInfo_c, &assocInfo, sizeof(array_info));
            
        // set up the cost matrix
        array_info spaceMapInfo, *spaceMapInfo_c;
        spaceMapInfo.rows = maximaInfo.rows;
        spaceMapInfo.cols = camInfo.rows;
        spaceMapInfo.elementSize = sizeof(float);
        safeCudaMalloc(&spaceMapInfo_c, sizeof(array_info));
        safeCudaMemcpy2Device(spaceMapInfo_c, &spaceMapInfo, sizeof(array_info));

        float* spaceTmp;
        safeCudaMalloc(&spaceTmp, maximaInfo.rows * camInfo.rows * sizeof(float));
        safeCudaMemset(spaceTmp, 0, maximaInfo.rows * camInfo.rows * sizeof(float));
        
        dim3 blockInfo(maximaInfo.rows,1);
        dim3 threadInfo(camInfo.rows, 1);

        setSpaceMap<<<blockInfo, threadInfo>>>(
            maxima.second,
            maximaInfo_c,
            camData,
            camInfo_cuda,
            spaceTmp,
            spaceMapInfo_c
        );
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            std::stringstream ss;
            ss << "setSpaceMapKernel failed\n";
            ss << cudaGetErrorString(error);
            throw std::runtime_error(ss.str());
        }  
        
        associateCameraKernel<<<1, threadInfo>>>(
            maxima.second,
            maximaInfo_c,
            camData,
            camInfo_cuda,
            associated,
            assocInfo_c,
            spaceTmp,
            spaceMapInfo_c,
            adjustFactor
        );
        cudaDeviceSynchronize();
        cudaError_t error2 = cudaGetLastError();
        if(error2 != cudaSuccess){
            std::stringstream ss;
            ss << "associateCameraKernel failed\n";
            ss << cudaGetErrorString(error2);
            throw std::runtime_error(ss.str());
        }
        
        joinFeatures<<<blockInfo, threadInfo>>>(
            maxima.second,
            maximaInfo_c,
            camData,
            camInfo_cuda,
            associated,
            assocInfo_c,
            spaceTmp,
            spaceMapInfo_c
        );
        cudaDeviceSynchronize();
        cudaError_t error3 = cudaGetLastError();
        if(error3 != cudaSuccess){
            std::stringstream ss;
            ss << "joinFeaturesKernel failed\n";
            ss << cudaGetErrorString(error3);
            throw std::runtime_error(ss.str());
        }

        safeCudaFree(spaceTmp);
        safeCudaFree(spaceMapInfo_c);
    
        // move the results back to host memory
        float* ret;
        ret = (float*)malloc(assocInfo.size());
        safeCudaMemcpy2Host(ret, associated, assocInfo.size());
        safeCudaFree(associated);
    
        // keep only the valid rows
        std::vector<float> retVec;
        retVec.reserve(assocInfo.rows * (assocInfo.cols-1)); // don't put isValid in the return vector
        for(size_t i = 0; i < assocInfo.rows; i++){
            if(ret[array_index_cpu(i, 5, &assocInfo)] != 1.0) continue;
            for(size_t j = 0; j < (assocInfo.cols-1); j++)
                retVec.push_back(ret[array_index_cpu(i, j, &assocInfo)]);
        }
    
        // save the data from the vector in a contiguous array
        //memset(ret, 0, assocInfo.size());
        memcpy(ret, retVec.data(), sizeof(float) * retVec.size());
        assocInfo.rows = retVec.size() / (assocInfo.cols-1);
        assocInfo.cols = assocInfo.cols-1;  // not isValid
    
        return std::pair<array_info,float*>(assocInfo,ret);
    }
}

void GaussMap::reset(){
    checkCudaError(cudaMemset(array, 0, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));
    
    // use thrust to set to windowIds to -1
    thrust::device_ptr<int16_t> dev_ptr(windowIds);
    size_t offset = windowIdInfo.rows * windowIdInfo.cols;
    thrust::fill(dev_ptr, dev_ptr+offset, (int16_t)(-1));
    setRadarIds();

    safeCudaFree(radarData);
    safeCudaFree(camData);

    radarData = nullptr;
    camData = nullptr;
}
#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif
#include <math_constants.h>     // CUDART_PI_F

__device__ __forceinline__
float calcPdf(float stdDev, float mean, float radius){
    // calculate the pdf of the given radar point based on the radius
    CUDA_ASSERT_POS_E(radius, "Cannot calculate pdf for radius < 0");
    CUDA_ASSERT_POS(stdDev, "standard deviation cannot be <= 0");
    constexpr float inv_sqrt_2pi = 0.3989422804014327;
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
Position_t indexDiff(size_t row, 
                     size_t col, 
                     const RadarData_t* __restrict__ radarData, 
                     size_t radarPointIdx, 
                     const array_info* __restrict__ radarInfo, 
                     const array_info* __restrict__ mapInfo, 
                     const array_rel* __restrict__ mapRel){
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
Position_t index_to_position(size_t row, size_t col, const array_info * __restrict__ info, const array_rel * __restrict__ relation){
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
void radarPointKernel(mapType_t* __restrict__ gaussMap, 
                      const RadarData_t* __restrict__ radarData, 
                      const array_info* __restrict__ mapInfo, 
                      const array_rel* __restrict__ mapRel, 
                      const array_info* __restrict__ radarInfo,
                      const distInfo_t* __restrict__ distributionInfo,
                      radarId_t* __restrict__ radarIds){
    // In this function, the radar point id is threadIdx.x
    radarId_u un;

    float stdDev = distributionInfo->stdDev;
    float adjustment = 1;//radarData[array_index(threadIdx.x, 4, radarInfo)];
    
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

        float pdfVal = calcPdf(stdDev, distributionInfo->mean, radius) * adjustment;
        CUDA_ASSERT_POS(pdfVal, "negative pdf value");

        atomicAdd(&gaussMap[array_index(blockIdx.x,col,mapInfo)], pdfVal);

        un.radData.radarId = threadIdx.x;
        un.radData.garbage = 0;
        un.radData.probability = pdfVal;
        CUDA_ASSERT_LT_E(threadIdx.x, 32767, "Too many radar ids this cycle");
        atomicMax((unsigned long long int*)&radarIds[array_index(blockIdx.x, col, mapInfo)], un.ulong);

    }
}
#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif

/* Functions like memset, but since cudaMemset takes an integer, 
 * this is necessary. This assigns an unsigned long long int to the
 * memory address of every element in the array. Since it's a kernel,
 * each thread does one operation */
__global__
void setRadarIdsKernel(radarId_t *array){
    array[blockIdx.x].radarId = -1;
    array[blockIdx.x].garbage = 0;
    array[blockIdx.x].probability = 0.0;
}

//-----------------------------------------------------------------------------
