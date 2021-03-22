#include "gaussMap.cuh"

__global__
void calcMaxKernel(maxVal_t *isMax, 
                  const float* array, 
                  const array_info *mapInfo,
                  const radarId_t *radarIds,
                  unsigned short windowSize){
    /*
    for every point in the map, check a 7x7 grid (up to the edges) to see if it
    is a local max. if it is, put the ids of each radar in this range into a list
    so the algorithm can use it later
    */
    int col = threadIdx.x;
    int row = blockIdx.x;
    if(row == 0 || row > mapInfo->rows) return;
    if(col == 0 || col > mapInfo->cols) return;
    
    float curVal = array[array_index(row,col, mapInfo)];
    if(curVal == 0) return; // not a max if it's zero

    maxVal_t *toInsert;
    toInsert = &isMax[array_index(row,col,mapInfo)];
    // for(size_t i = 0; i < )

    size_t iterator = 0;
    for(int i = -windowSize; i <= windowSize; i++){
        for(int j = -windowSize; j <= windowSize; j++){
            if(row+i > mapInfo->rows || col+j > mapInfo->cols) {
                toInsert->radars[iterator++] = -1;
                continue;
            }
            
            if(array[array_index(row+i, col+j, mapInfo)] > curVal){
                toInsert->isMax = 0;
                for(;iterator < searchSize; iterator++)
                    toInsert->radars[iterator] = -1;
                return;
            }
            
            if(row+i >= 0 && col+j >= 0)
                toInsert->radars[iterator++] = radarIds[array_index(row+i, col+j, mapInfo)].radarId;
        }
    }

    toInsert->isMax = 1;
    toInsert->classVal = 0;
}

__global__ 
void aggregateMax(const mapType_t *array, 
                  const array_info *mapInfo, 
                  const array_rel *mapRel,
                  const maxVal_t *isMax,
                  float* ret, 
                  const array_info* maxInfo,
                  const RadarData_t *radarData, 
                  const array_info *radarInfo, 
                  const int *maximaLocs,
                  const array_info *locsInfo)
{
    // creates an array with the return information in the form of:
    // [row, col, pdfVal, vx, vy, targetId]
    int row,col;
    row = maximaLocs[array_index(threadIdx.x, 0, locsInfo, __LINE__)];
    col = maximaLocs[array_index(threadIdx.x, 1, locsInfo, __LINE__)];
    
    maxVal_t tmp = isMax[row * mapInfo->cols + col];    

    ret[array_index(threadIdx.x, 0, maxInfo,__LINE__)] = (col - mapInfo->cols/2.0) / mapRel->res;
    ret[array_index(threadIdx.x, 1, maxInfo,__LINE__)] = -(row - mapInfo->rows/2.0) / mapRel->res;
    ret[array_index(threadIdx.x, 2, maxInfo,__LINE__)] = array[array_index(row, col, mapInfo,__LINE__)];
    ret[array_index(threadIdx.x, 3, maxInfo,__LINE__)] = calcMean(2, tmp.radars, radarData, radarInfo);
    ret[array_index(threadIdx.x, 4, maxInfo,__LINE__)] = calcMean(3, tmp.radars, radarData, radarInfo);

    // attach the target ID to the maxima.
    // we assume there is only one target for each maxima detection
    int targetId = -1;
    int tmpId;
    for(size_t i = 0; i < searchSize; i++){
        if(tmp.radars[i] == -1) continue;
        tmpId = radarData[array_index(tmp.radars[i], 5, radarInfo,__LINE__)];
        if(tmpId == -1) continue;

        targetId = tmpId;
    }
    ret[array_index(threadIdx.x, 5, maxInfo,__LINE__)] = targetId;
}

__device__
float calcMean(size_t col, 
               const int16_t* radars, 
               const RadarData_t *radarData, 
               const array_info *radarInfo)
{
    float total = 0.0f;
    size_t numPoints = 0;
    for(size_t i = 0; i < searchSize; i++){
        if(radars[i] == -1) continue;

        CUDA_ASSERT_LT_E(radars[i], radarInfo->rows, "Radar number larger than number of rows");

        total += radarData[array_index(radars[i], col, radarInfo, __LINE__)];
        numPoints++;
    }

    // divide by zero is bad. But apparently it's zero!
    if(numPoints == 0)
        return 0.0f;
    
    return (total/numPoints);
}
