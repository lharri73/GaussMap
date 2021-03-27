#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif
#include <sstream>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

// allocate this struct in shared memory so we don't have to copy
// it to each kernel when it's needed

// expected to be in: [x,y,vx,vy,wExist,targetId]
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
    assocInfo.cols = 7; // [x,y,vx,vy,class,isValid]
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
            if(ret[i*assocInfo.cols + 5] == 0.0){
                continue;
            }else{
                for(size_t j = 0; j < (assocInfo.cols-1); j++)
                    retVec.push_back(ret[i*assocInfo.cols + j]);
            }
        }
    
        // save the data from the vector in a contiguous array
        memset(ret, 0, assocInfo.size());
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
