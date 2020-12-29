#include <sstream>
#include "gaussMap.cuh"

__device__
size_t array_index(size_t row, size_t col, array_info info){
    return row * info.rows + col;
}

// find the position from center of map given cell index
// ret: dim3 (x,y,radius)
__device__ 
dim3 index_to_position(size_t row, size_t col, array_info info){
    return dim3(0,0,0);
}


__global__ 
void radarPointKernel(mapType_t* gaussMap, RadarData_t *radarData, array_info *mapInfo, array_rel* mapRel, array_info* radarInfo){
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
    checkCudaError(cudaMemcpy(mapInfo_cuda, tmpa, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarInfo_cuda, tmpb, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mapRel_cuda, tmpc, sizeof(struct Array_Relationship), cudaMemcpyHostToDevice));

    free(tmpa);
    free(tmpb);
    free(tmpc);


    // dispatch the kernel with `numPoints` threads
    radarPointKernel<<<1,numPoints>>>(
        array,
        radarData,
        mapInfo_cuda,
        mapRel_cuda,
        radarInfo_cuda
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
