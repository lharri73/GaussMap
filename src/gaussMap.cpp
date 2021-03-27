#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.hpp"
#else
#include "gaussMap.hpp"
#endif

void GaussMap::init(int mapHeight, int mapWidth, int mapResolution, bool useMin){
    // Set parameters
    mapRel.height = mapHeight;
    mapRel.width = mapWidth;
    mapRel.res = mapResolution;
    mapInfo.cols = mapRel.width * mapRel.res;
    mapInfo.rows = mapRel.height * mapRel.res;
    mapInfo.elementSize = sizeof(mapType_t);
    windowIdInfo.rows = mapInfo.rows * mapInfo.cols;
    windowIdInfo.cols = searchSize;
    windowIdInfo.elementSize = sizeof(int16_t);
    if(!useMin) minCutoff = std::numeric_limits<float>::min();

    // allocate memory
    safeCudaMalloc(&array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize);
    safeCudaMalloc(&radarIds, sizeof(unsigned long long int) * mapInfo.rows * mapInfo.cols);
    safeCudaMalloc(&windowIds, windowIdInfo.rows * windowIdInfo.cols * sizeof(int16_t));

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    safeCudaMalloc(&mapInfo_cuda, sizeof(array_info));
    safeCudaMalloc(&radarInfo_cuda, sizeof(array_info));
    safeCudaMalloc(&camInfo_cuda, sizeof(array_info));
    safeCudaMalloc(&windowIdInfo_cuda, sizeof(array_info));
    safeCudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship));
    safeCudaMalloc(&radarDistri_c, sizeof(distInfo_t));
    safeCudaMemcpy2Device(mapInfo_cuda, &mapInfo, sizeof(array_info));
    safeCudaMemcpy2Device(mapRel_cuda, &mapRel, sizeof(struct Array_Relationship));
    safeCudaMemcpy2Device(radarDistri_c, radarDistri, sizeof(distInfo_t));
    safeCudaMemcpy2Device(windowIdInfo_cuda, &windowIdInfo, sizeof(array_info));
    reset();
    printf("here\n");
}

GaussMap::~GaussMap(){
    // this destructor is called last
    safeCudaFree(array);
    safeCudaFree(mapInfo_cuda);
    safeCudaFree(radarInfo_cuda);
    safeCudaFree(camInfo_cuda);
    safeCudaFree(mapRel_cuda);
    safeCudaFree(radarIds);
    safeCudaFree(radarData);
    safeCudaFree(camData);
    free(radarDistri);
    safeCudaFree(radarDistri_c);
    safeCudaFree(windowIdInfo_cuda);
}
