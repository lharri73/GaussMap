#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>

GaussMap::GaussMap(int Width, int Height, int Cell_res){
    mapInfo.rows = Height * Cell_res;
    mapInfo.cols = Width * Cell_res;
    // allocate memory for the array
    cudaError_t error = cudaMallocPitch(&array, &mapInfo.pitch, mapInfo.cols, mapInfo.rows);
    checkCudaError(error);
    error = cudaMemset2D(array, mapInfo.pitch, 0, mapInfo.cols, mapInfo.rows);
    checkCudaError(error);

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
        cudaError_t error = cudaFree(array);
        checkCudaError(error);
        if(radarData != nullptr){
            error = cudaFree(radarData);
            checkCudaError(error);
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
    checkCudaError(error);

    error = cudaMemcpy(radarData, data, sizeof(RadarData_t) * numPoints * radarFeatures, cudaMemcpyHostToDevice);
    checkCudaError(error);

    calcRadarMap();
}


PYBIND11_MODULE(gaussMap, m){
    // m.def("multiply_with_scalar", multiply_with_scalar);
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData);
}
