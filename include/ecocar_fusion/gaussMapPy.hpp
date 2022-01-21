#pragma once
#include "gaussMap.hpp"
#include "types.hpp"
#include "params.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>
namespace py = pybind11;

class GaussMapPy : public GaussMap{
    public:
        GaussMapPy(const std::string params);
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);
        void addCameraData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);
        // returns the heatmap as a 2 dimensional numpy array
        py::array_t<mapType_t> asArray();
        py::array_t<float> findMax();
        py::array_t<uint16_t> classes();
        py::array_t<float> associate();
};
