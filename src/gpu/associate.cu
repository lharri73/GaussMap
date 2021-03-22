#include "gaussMap.cuh"

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
    const array_info *spaceMapInfo
){
    /*
    radarData: [row, col, class, pdfVal, vx, vy]
    cameraData: [x,y,class]
    ret: [x,y,vx,vy,class,isValid]
    */
   
    // we only need one because we need to find the max of the threadIdx.xumn vector
    
    float min = 3e38;// This is near the max value of a float //spaceMap[array_index(0, threadIdx.x, &spaceMapInfo)];
    float cur;
    int minIndex = -1;

    // find the closest radar point
    for(size_t i = 0; i < spaceMapInfo->rows; i++){
        cur = spaceMap[array_index(i, threadIdx.x, spaceMapInfo)];
        if(cur < min){
            min = cur;
            minIndex = (int)i;
        }
    }

    if(minIndex > -1 && min <= 2.0){
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
        results[array_index(blockIdx.x, 2, resultInfo)] = radarData[array_index(row, 4, radarInfo)]; //vx
        results[array_index(blockIdx.x, 3, resultInfo)] = radarData[array_index(row, 5, radarInfo)]; //vy
        results[array_index(blockIdx.x, 4, resultInfo)] = camData[array_index(col, 2, camInfo)];     //class
        results[array_index(blockIdx.x, 5, resultInfo)] = 1;                                         //isValid
        return;
    }

    if(threadIdx.x == 0 && results[array_index(blockIdx.x, 5, resultInfo)] != 1.0){
        // this is just a radar detection
        results[array_index(blockIdx.x, 0, resultInfo)] = radarData[array_index(row, 0, radarInfo)]; //x
        results[array_index(blockIdx.x, 1, resultInfo)] = radarData[array_index(row, 1, radarInfo)]; //y
        results[array_index(blockIdx.x, 2, resultInfo)] = radarData[array_index(row, 4, radarInfo)]; //vx
        results[array_index(blockIdx.x, 3, resultInfo)] = radarData[array_index(row, 5, radarInfo)]; //vy
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