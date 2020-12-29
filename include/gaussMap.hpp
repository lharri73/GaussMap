#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <cuda_runtime.h>
namespace py = pybind11;

typedef double RadarData_t;
typedef float mapType_t;

typedef struct Array_Info{
    size_t rows;
    size_t cols;
    size_t elementSize;
} array_info;

typedef struct Array_Relationship{
    size_t width;   // meters
    size_t height;  // meters
    size_t res;     // resolution (cells per meter)
} array_rel;

class GaussMap{
    private:
        mapType_t* array;
        array_info mapInfo, *mapInfo_cuda;
        array_rel mapRel, *mapRel_cuda;

        RadarData_t* radarData; // set to nullptr until received
        array_info radarInfo, *radarInfo_cuda;
        bool allClean;

        int cell_res;

        // radar point info
        // populated after addRadarData called
        size_t numPoints;
        size_t radarFeatures;

        void calcRadarMap();
    public:
        GaussMap(int width, int height, int cell_res);
        ~GaussMap();
        void cleanup();
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);
        py::array_t<mapType_t> asArray();

};
