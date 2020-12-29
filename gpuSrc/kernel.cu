#include <sstream>
#include "gaussMap.cuh"

__device__
size_t array_index(size_t row, size_t col, array_info *info){
    // helper function to find the array index
    return row * info->rows + col;
}

template <typename T>
__device__
double radiusFromPos(T x, T y){
    // return the radius from the position from origin (at center of map)
    // x and y in meters
    return hypot((double)x, (double)y);
}

__device__ 
dim3 index_to_position(size_t row, size_t col, array_info *info, array_rel *relation){
    // find the position from center of map given cell index
    // ret: dim3 (x,y,radius)
    float center_x = (float)(info->cols/2.0);
    float center_y = (float)(info->rows/2.0);
    float x_offset = col - center_x;
    float y_offset = (row - center_y) * -1;     // flip the y axis so + is in the direction of travel

    dim3 ret;
    ret.x = x_offset / relation->res;
    ret.y = y_offset / relation->res;
    ret.z = radiusFromPos(ret.x, ret.y);

    return ret;
}



__global__ 
void radarPointKernel(mapType_t* gaussMap, 
                      RadarData_t *radarData, 
                      array_info *mapInfo, 
                      array_rel* mapRel, 
                      array_info* radarInfo,
                      double* distributionInfo){
    // printf("here!\n");
}

void GaussMap::calcRadarMap(){

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    array_info *tmpa, *tmpb;
    array_rel *tmpc;
    tmpa = (array_info*)malloc(sizeof(struct Array_Info));
    tmpb = (array_info*)malloc(sizeof(struct Array_Info));
    tmpc = (array_rel*)malloc(sizeof(struct Array_Relationship));
    memcpy(tmpa, &mapInfo, sizeof(struct Array_Info));
    memcpy(tmpb, &radarInfo, sizeof(struct Array_Info));
    memcpy(tmpc, &mapRel, sizeof(struct Array_Relationship));

    checkCudaError(cudaMalloc(&mapInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&radarInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship)));
    checkCudaError(cudaMalloc(&radarDistri_c, 2*sizeof(double)));
    checkCudaError(cudaMemcpy(mapInfo_cuda, tmpa, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarInfo_cuda, tmpb, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mapRel_cuda, tmpc, sizeof(struct Array_Relationship), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarDistri_c, radarDistri, 2*sizeof(double), cudaMemcpyHostToDevice));

    free(tmpa);
    free(tmpb);
    free(tmpc);


    // dispatch the kernel with `numPoints` threads
    radarPointKernel<<<1,numPoints>>>(
        array,
        radarData,
        mapInfo_cuda,
        mapRel_cuda,
        radarInfo_cuda,
        radarDistri_c
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "radarPointKernel launch failed\n";
        ss << cudaGetErrorString(error);
        throw std::string(ss.str());
    }

    // wait untill all threads sync
    cudaDeviceSynchronize();
}
