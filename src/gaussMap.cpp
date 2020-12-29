#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>

// allocates memory for the map
GaussMap::GaussMap(int Width, int Height, int Cell_res){
    mapInfo.rows = Height * Cell_res;
    mapInfo.cols = Width * Cell_res;
    mapInfo.elementSize = sizeof(mapType_t);

    mapRel.width = Width;
    mapRel.height = Height;
    mapRel.res = Cell_res;
    // allocate memory for the array
    cudaError_t error = cudaMalloc(&array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize);
    checkCudaError(error);
    error = cudaMemset(array, 0, mapInfo.cols * mapInfo.rows * mapInfo.elementSize);
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

// performs the cleanup steps. Frees memory
void GaussMap::cleanup(){
    if(!allClean){
        cudaError_t error = cudaFree(array);
        checkCudaError(error);

        error = cudaFree(mapInfo_cuda);
        checkCudaError(error);
        if(radarData != nullptr){
            error = cudaFree(radarData);
            checkCudaError(error);
            error = cudaFree(radarInfo_cuda);
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

    numPoints = buf1.shape[0];      // num points
    radarFeatures = buf1.shape[1];          // usually 18
    if(radarFeatures != 18){
        throw std::runtime_error("Got invalid shape of Radar Data. should be Nx18");
    }
    radarInfo.elementSize = sizeof(RadarData_t);
    radarInfo.cols = radarFeatures;
    radarInfo.rows = numPoints;

    // allocate and copy the array to the GPU so we can run a kernel on it
    cudaError_t error = cudaMalloc(&radarData, sizeof(RadarData_t) * numPoints * radarFeatures);
    checkCudaError(error);

    error = cudaMemcpy(radarData, data, sizeof(RadarData_t) * numPoints * radarFeatures, cudaMemcpyHostToDevice);
    checkCudaError(error);

    calcRadarMap();     // setup for the CUDA kernel. in GPU code
}

// returns numpy array to python of gaussMap
py::array_t<mapType_t> GaussMap::asArray(){
    mapType_t* retArray;
    retArray = (mapType_t*)malloc(sizeof(mapType_t) * mapInfo.cols * mapInfo.rows);

    cudaError_t error = cudaMemcpy(retArray, array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize, cudaMemcpyDeviceToHost);
    checkCudaError(error);

    py::buffer_info a(
        retArray, 
        sizeof(mapType_t), 
        py::format_descriptor<mapType_t>::format(), 
        2, 
        {mapInfo.rows, mapInfo.cols},
        {sizeof(mapType_t) * mapInfo.cols, sizeof(mapType_t) * 1});
    
    return py::array_t<mapType_t>(a);
}

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData)
        .def("asArray", &GaussMap::asArray);
}
