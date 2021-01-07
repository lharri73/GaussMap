#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>

#include "types.hpp"

namespace py = pybind11;

class GaussMap{
    public:
        GaussMap(const std::string params);
        ~GaussMap();        
        
        // used to add radar data to the map. (can only be called once) TODO: make sure only called once
        // takes a contiguous, 2 dimensional numpy array
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);
        void addCameraData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);

        // returns the heatmap as a 2 dimensional numpy array
        py::array_t<mapType_t> asArray();
        py::array_t<float> findMax();
        py::array_t<uint16_t> classes();
        
        void reset();

    private:
        mapType_t* array;
        array_info mapInfo, *mapInfo_cuda;
        array_rel mapRel, *mapRel_cuda;

        RadarData_t* radarData;     // set to nullptr until received
        array_info radarInfo, *radarInfo_cuda;

        float* cameraData;          // set to nullptr until received
        camVal_t* cameraClassData;
        array_info cameraInfo, *cameraInfo_cuda;
        array_info camClassInfo, *camClassInfo_cuda;

        std::vector<distInfo_t> cameraDistri;
        distInfo_t* cameraDistri_c;

        distInfo_t* radarDistri;         // normal distrubution info. 
        distInfo_t* radarDistri_c;       // 0: stddev, 1: mean, 2: distance cutoff

        void calcRadarMap();        // function used to setup the kernel. 
                                    // called from addRadarData()

        void calcCameraMap();       // function used to setup the kernel. 
                                    // called from addCameraData()

        std::vector<float> calcMax();
        float minCutoff;
        bool useMin;
};
