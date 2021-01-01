#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

namespace py = pybind11;

typedef float RadarData_t;
typedef float mapType_t;

typedef struct Array_Info{
    size_t rows;            // total number of rows
    size_t cols;            // total number of columns
    size_t elementSize;     // size of a single element in bytes
} array_info;

typedef struct Array_Relationship{
    size_t width;           // meters
    size_t height;          // meters
    size_t res;             // resolution (cells per linear meter)
} array_rel;

class GaussMap{
    private:
        mapType_t* array;
        array_info mapInfo, *mapInfo_cuda;
        array_rel mapRel, *mapRel_cuda;

        RadarData_t* radarData; // set to nullptr until received
        array_info radarInfo, *radarInfo_cuda;

        float *arrayPrime, *arrayPrimePrime;        //first and second derivative
        array_info primeInfo, *primeInfo_cuda;
        array_info primePrimeInfo, *primePrimeInfo_cuda;

        bool allClean;

        float* radarDistri;    // normal distrubution info. 
        float* radarDistri_c;  // 0: stddev, 1: mean, 2: distance cutoff

        // radar point info
        // populated after addRadarData called
        size_t numPoints;
        size_t radarFeatures;

        void calcRadarMap();        // function used to setup the kernel. 
                                    // called from addRadarData()

        void calcDerivative();
    public:
        GaussMap(int width, int height, int cell_res, double radarStdDev, double radarMean, double radarCutoff);
        
        // destructor functions
        ~GaussMap();
        void cleanup();
        
        // used to add radar data to the map. (can only be called once) TODO: make sure only called once
        // takes a contiguous, 2 dimensional numpy array
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);

        // returns the heatmap as a 2 dimensional numpy array
        py::array_t<mapType_t> asArray();
        std::vector<py::array_t<float> > derivative();
};
