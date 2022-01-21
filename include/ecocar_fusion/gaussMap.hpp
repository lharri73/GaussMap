#pragma once
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <thread>
#include <iostream>
#include <limits>
#include <mutex>
#include <algorithm>

#ifdef NUSCENES
#include "types.hpp"
#include "cudaUtils.hpp"
#include "params.hpp"
#else
#include "ecocar_fusion/types.hpp"
#include "ecocar_fusion/cudaUtils.hpp"
#include "ecocar_fusion/params.hpp"
#endif


class GaussMap{
    public:
        std::pair<array_info,float*> associatePair(); // function fusing the radar and vision detections
        void reset();                       // clears the array, radar, and camera data for the next fusion cycle
        ~GaussMap();
        void init(int mapHeight, int mapWidth, int mapResolution, bool useMin);
        
    protected:
        mapType_t* array;                   // cuda array for heatmap of gauss distributions
        array_info mapInfo, *mapInfo_cuda;  // holds size of heatmap (cuda and host)
        array_rel mapRel, *mapRel_cuda;     // holds resolution and real world size (m) of heatmap

        RadarData_t* radarData = nullptr;   // cuda array for radar data. set to nullptr until received
        array_info radarInfo, *radarInfo_cuda;
        radarId_t *radarIds;
        int16_t *windowIds;                 // cuda radar ids involved in each window during a maxima calculation
        array_info windowIdInfo, *windowIdInfo_cuda;// holds size of radar ids

        float* camData = nullptr;           // cuda array for camera data. set to nullptr until eceivedr
        array_info camInfo, *camInfo_cuda;  // holds size of camera data (rows,cols,element size (sizeof(float)))

        distInfo_t* radarDistri;            // normal distrubution info. 
        distInfo_t* radarDistri_c;          // 0: stddev, 1: mean, 2: distance cutoff


        std::mutex runLock;                 // mutex to prevent arrays being overwritten during processing

        float minCutoff = 0;                // parameter used during pdf calculation
        bool useMin = false;                // whether this parameter is used or not (sets above to min of float (-inf))
        float adjustFactor = 1;             // factor for distance to neighbor association search
        
        void calcRadarMap();                // function used to setup the kernel. 
                                            // called from addRadarData()
        std::pair<array_info,float*> calcMax(); // calculates the local maxima of GaussMap::array
        void setRadarIds();                 // resets the radarIds grid to -1 during reset cycle
};

