// #include <iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "gaussMap.hpp"
#include <iostream>

GaussMap::GaussMap(int Width, int Height, int Vcells, int Hcells) : 
    height{Height}, width{Width}, vcells{Vcells}, hcells{Hcells}{
    // allocate memory for the array
    cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKind::cudaChannelFormatKindSigned;
    desc.x = 8; // use 8 bits for the x fields
    // dont use the y,z,w fields
    desc.y = 0;
    desc.z = 0;
    desc.w = 0;

    cudaError_t error = cudaMallocArray(&array, &desc, width, height);
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }

    radarData = nullptr;

    allClean = false;
}

GaussMap::~GaussMap(){
    // there isn't a nice way to call destructors from 
    // python, so we do it this way. 
    if(!allClean)
        cleanup();
}

void GaussMap::cleanup(){
    if(!allClean){
        cudaError_t error = cudaFreeArray(array);
        if(error != cudaSuccess){
            throw std::runtime_error(cudaGetErrorString(error));
        }
        if(radarData != nullptr){
            error = cudaFree(radarData);
            if(error != cudaSuccess){
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    }

    allClean = true;
}

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMap::addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array){
    // get information about the numpy array from python
    py::buffer_info buf1 = array.request();
    RadarData_t *data;
    data = static_cast<RadarData_t*>(buf1.ptr);
    if(buf1.itemsize != sizeof(RadarData_t)){
        throw std::runtime_error("Invalid datatype passed with radar data. Should be type: float (float32).");
    }

    numPoints = buf1.shape[1];
    radarFeatures = buf1.shape[0]; // usually 18

    // allocate and copy the array to the GPU so we can run a kernel on it
    cudaError_t error = cudaMalloc(&radarData, sizeof(RadarData_t) * numPoints * radarFeatures);
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaMemcpy(radarData, data, sizeof(RadarData_t) * numPoints * radarFeatures, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }
    calcRadarMap();
}


PYBIND11_MODULE(gaussMap, m){
    // m.def("multiply_with_scalar", multiply_with_scalar);
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int,int>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData);
}
