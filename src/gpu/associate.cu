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
    const RadarData_t *radarData,
    const array_info *radarInfo,
    const float* camData,
    const array_info *camInfo,
    float* spaceMap,
    const array_info *spaceMapInfo)
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
    const RadarData_t *radarData,
    const array_info *radarInfo,
    const float* camData,
    const array_info *camInfo,
    float* results,
    const array_info *resultInfo,
    float* spaceMap,
    const array_info *spaceMapInfo,
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
    const RadarData_t *radarData,
    const array_info *radarInfo,
    const float* camData,
    const array_info *camInfo,
    float* results,
    const array_info *resultInfo,
    float* spaceMap,
    const array_info* spaceMapInfo
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
void singleElementResult(const float* radarData,
                         const array_info *radarInfo,
                         const float* camData,
                         const array_info *camInfo,
                         float* results,
                         const array_info *resultInfo)
{
    const float *array;
    const array_info *info;
    if(camInfo->rows == 0){
        array = radarData;
        info = radarInfo;
    }
    else{
        array = camData;
        info = camInfo;
    }

    results[array_index(blockIdx.x, 0, resultInfo)] = array[array_index(blockIdx.x, 0, info)];
    results[array_index(blockIdx.x, 1, resultInfo)] = array[array_index(blockIdx.x, 1, info)];
    results[array_index(blockIdx.x, 2, resultInfo)] = camInfo->rows == 0 ? array[array_index(blockIdx.x, 4, info)] : 0.0;
    results[array_index(blockIdx.x, 3, resultInfo)] = camInfo->rows == 0 ? array[array_index(blockIdx.x, 5, info)] : 0.0;
    results[array_index(blockIdx.x, 4, resultInfo)] = camInfo->rows == 0 ? 0.0 : array[array_index(blockIdx.x, 2, info)];
}

//-----------------------------------------------------------------------------
// mergeSort
// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
__device__
void merge(valInt_t arr[], int l, int m, int r);
 
/* Iterative mergesort function to sort arr[0...n-1] */
__device__
void mergeSort(valInt_t arr[], int n)
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
