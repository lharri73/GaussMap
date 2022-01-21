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
