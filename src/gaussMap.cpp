#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>

// allocates memory for the map
GaussMap::GaussMap(const std::string params){
    YAML::Node config = YAML::LoadFile(params);

    mapRel.height = config["MapHeight"].as<int>();
    mapRel.width = config["MapWidth"].as<int>();
    mapRel.res = config["MapResolution"].as<int>();

    mapInfo.cols = mapRel.width * mapRel.res;
    mapInfo.rows = mapRel.height * mapRel.res;
    mapInfo.elementSize = sizeof(mapType_t);
    // allocate memory for the array
    checkCudaError(cudaMalloc(&array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));
    checkCudaError(cudaMemset(array, 0, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));

    radarDistri = (float*)calloc(3, sizeof(float));
    radarDistri[0] = config["Radar"]["StdDev"].as<float>();
    radarDistri[1] = config["Radar"]["Mean"].as<float>();
    radarDistri[2] = config["Radar"]["RadCutoff"].as<float>();

    cameraDistri = (float*)calloc(3, sizeof(float));
    cameraDistri[0] = config["Camera"]["StdDev"].as<float>();
    cameraDistri[1] = config["Camera"]["Mean"].as<float>();
    cameraDistri[2] = config["Camera"]["RadCutoff"].as<float>();

    // this is all done so we can check if it has been allocated later
    radarData = nullptr;
    cameraData = nullptr;
    cameraClassData = nullptr;

    mapInfo_cuda = nullptr;
    mapRel_cuda = nullptr;
    radarInfo_cuda = nullptr;
    cameraInfo_cuda = nullptr;
    camClassInfo_cuda = nullptr;

    radarDistri_c = nullptr;
    cameraDistri_c = nullptr;

    allClean = false;
}

GaussMap::~GaussMap(){
    if(!allClean)
        cleanup();
}

void GaussMap::cleanup(){
    // performs the cleanup steps. Frees memory
    if(!allClean){
        safeCudaFree(array);
        safeCudaFree(mapInfo_cuda);
        safeCudaFree(mapRel_cuda);

        if(radarData != nullptr){
            checkCudaError(cudaFree(radarData));
            safeCudaFree(radarInfo_cuda);
        }
        if(cameraData != nullptr){
            checkCudaError(cudaFree(cameraData));
            safeCudaFree(cameraInfo_cuda);
        }
        free(radarDistri);
        free(cameraDistri);

        safeCudaFree(cameraDistri_c);
        safeCudaFree(radarDistri_c);
        safeCudaFree(cameraClassData);       
        safeCudaFree(camClassInfo_cuda);

    }
    allClean = true;
}

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMap::addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array){
    size_t radarPoints, radarFeatures;
    
    // get information about the numpy array from python
    py::buffer_info buf1 = array.request();
    RadarData_t *data;
    data = static_cast<RadarData_t*>(buf1.ptr);
    if(buf1.itemsize != sizeof(RadarData_t)){
        throw std::runtime_error("Invalid datatype passed with radar data. Should be type: float (float32).");
    }

    radarPoints = buf1.shape[0];      // num points
    radarFeatures = buf1.shape[1];          // usually 18
    if(radarFeatures != 18){
        throw std::runtime_error("Got invalid shape of Radar Data. should be Nx18");
    }
    radarInfo.elementSize = sizeof(RadarData_t);
    radarInfo.cols = radarFeatures;
    radarInfo.rows = radarPoints;

    // allocate and copy the array to the GPU so we can run a kernel on it
    checkCudaError(cudaMalloc(&radarData, sizeof(RadarData_t) * radarPoints * radarFeatures));

    checkCudaError(cudaMemcpy(radarData, data, sizeof(RadarData_t) * radarPoints * radarFeatures, cudaMemcpyHostToDevice));

    calcRadarMap();     // setup for the CUDA kernel. in GPU code
}

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMap::addCameraData(py::array_t<float, py::array::c_style | py::array::forcecast> array){
    size_t numPoints, numFeatures;

    // get information about the numpy array from python
    py::buffer_info buf1 = array.request();
    float *data;
    data = static_cast<float*>(buf1.ptr);
    if(buf1.itemsize != sizeof(float)){
        throw std::runtime_error("Invalid datatype passed with camera data. Should be type: float (float32).");
    }

    numPoints = buf1.shape[0];      // num points
    numFeatures = buf1.shape[1];          // usually 3
    if(numFeatures != 3){
        throw std::runtime_error("Got invalid shape of camera Data. should be Nx3");
    }
    cameraInfo.elementSize = sizeof(float);
    cameraInfo.cols = numFeatures;
    cameraInfo.rows = numPoints;

    // allocate and copy the array to the GPU so we can run a kernel on it
    checkCudaError(cudaMalloc(&cameraData, sizeof(float) * numPoints * numFeatures));

    checkCudaError(cudaMemcpy(cameraData, data, sizeof(float) * numPoints * numFeatures, cudaMemcpyHostToDevice));

    calcCameraMap();     // setup for the CUDA kernel. in GPU code
}

// returns numpy array to python of gaussMap
py::array_t<mapType_t> GaussMap::asArray(){
    mapType_t* retArray = new mapType_t[mapInfo.cols * mapInfo.rows];

    checkCudaError(cudaMemcpy(retArray, array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize, cudaMemcpyDeviceToHost));

    py::buffer_info a(
        retArray, 
        sizeof(mapType_t), 
        py::format_descriptor<mapType_t>::format(), 
        2, 
        {mapInfo.rows, mapInfo.cols},
        {sizeof(mapType_t) * mapInfo.cols, sizeof(mapType_t) * 1});
    
    return py::array_t<mapType_t>(a);
}

// return the indices of the local maxima of the gaussMap
// Nx2 [[row,col],...]
py::array_t<uint16_t> GaussMap::findMax(){

    std::vector<uint16_t> values = calcMax();
    uint16_t *vecData = (uint16_t*)malloc(values.size() * sizeof(uint16_t));
    memcpy(vecData, values.data(), values.size() * sizeof(uint16_t));

    int rows = values.size() /2;

    py::buffer_info ret(
        vecData,
        sizeof(uint16_t),
        py::format_descriptor<uint16_t>::format(),
        2,
        {rows,2},
        {sizeof(uint16_t) * 2, sizeof(uint16_t) * 1}
    );
    return py::array_t<uint16_t>(ret);
}

py::array_t<uint16_t> GaussMap::classes(){

    camVal_t *tmp = (camVal_t*)malloc(sizeof(camVal_t) * mapInfo.cols * mapInfo.rows);
    checkCudaError(cudaMemcpy(tmp, cameraClassData, sizeof(camVal_t) * mapInfo.cols * mapInfo.rows, cudaMemcpyDeviceToHost));
    uint16_t *data = (uint16_t*)malloc(sizeof(uint16_t) * mapInfo.cols * mapInfo.rows);

    for(size_t i = 0; i < mapInfo.rows; i++){
        for(size_t j = 0; j < mapInfo.cols; j++){
            data[i * mapInfo.cols + j] = tmp[i*mapInfo.cols + j].classVal;
        }
    }

    py::buffer_info ret(
        data,
        sizeof(uint16_t),
        py::format_descriptor<uint16_t>::format(),
        2,
        {mapInfo.rows, mapInfo.cols},
        {sizeof(uint16_t) * mapInfo.cols, sizeof(uint16_t) * 1}
    );

    return py::array_t<uint16_t>(ret);
}

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<std::string>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData)
        .def("addCameraData", &GaussMap::addCameraData)
        .def("asArray", &GaussMap::asArray, py::return_value_policy::take_ownership)
        .def("findMax", &GaussMap::findMax, py::return_value_policy::take_ownership)
        .def("classes", &GaussMap::classes, py::return_value_policy::take_ownership);
}
