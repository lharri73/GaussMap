#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>
#include <limits>

// allocates memory for the map
GaussMap::GaussMap(const std::string params){
    YAML::Node config = YAML::LoadFile(params);

    mapRel.height = config["MapHeight"].as<int>();
    mapRel.width = config["MapWidth"].as<int>();
    mapRel.res = config["MapResolution"].as<int>();

    useMin = config["UseMinValue"].as<bool>();
    minCutoff = config["MinGaussValue"].as<float>();
    if(!useMin)
        minCutoff = std::numeric_limits<float>::min();

    mapInfo.cols = mapRel.width * mapRel.res;
    mapInfo.rows = mapRel.height * mapRel.res;
    mapInfo.elementSize = sizeof(mapType_t);

    safeCudaMalloc(&array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize);
    // allocate memory for the radar ids
    safeCudaMalloc(&radarIds, sizeof(unsigned long long int) * mapInfo.rows * mapInfo.cols);

    radarDistri = (distInfo_t*)malloc(sizeof(struct DistributionInfo));
    radarDistri->stdDev = config["Radar"]["StdDev"].as<float>();
    radarDistri->mean = config["Radar"]["Mean"].as<float>();
    radarDistri->distCutoff = config["Radar"]["RadCutoff"].as<float>();

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    safeCudaMalloc(&mapInfo_cuda, sizeof(struct Array_Info));
    safeCudaMalloc(&radarInfo_cuda, sizeof(struct Array_Info));
    safeCudaMalloc(&camInfo_cuda, sizeof(struct Array_Info));
    safeCudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship));
    safeCudaMalloc(&radarDistri_c, sizeof(distInfo_t));
    
    safeCudaMemcpy2Device(mapInfo_cuda, &mapInfo, sizeof(struct Array_Info));
    safeCudaMemcpy2Device(mapRel_cuda, &mapRel, sizeof(struct Array_Relationship));
    safeCudaMemcpy2Device(radarDistri_c, radarDistri, sizeof(distInfo_t));

    // this is all done so we can check if it has been allocated later
    radarData = nullptr;
    camData = nullptr;
    returned = nullptr;
    reset();
}

GaussMap::~GaussMap(){
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
}

void GaussMap::reset(){
    checkCudaError(cudaMemset(array, 0, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));
    setRadarIds();

    safeCudaFree(radarData);
    safeCudaFree(camData);

    radarData = nullptr;
    camData = nullptr;
}
