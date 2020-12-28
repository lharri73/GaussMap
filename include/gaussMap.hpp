#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib>
namespace py = pybind11;

typedef double RadarData_t;

class GaussMap{
    private:
        cudaArray_t array;
        RadarData_t* radarData; // set to nullptr until received
        bool allClean;

        int height, width;
        int vcells, hcells;

        // radar point info
        // populated after addRadarData called
        size_t numPoints;
        size_t radarFeatures;

        void calcRadarMap();
    public:
        GaussMap(int width, int height, int vcells, int hcells);
        ~GaussMap();
        void cleanup();
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);

};

