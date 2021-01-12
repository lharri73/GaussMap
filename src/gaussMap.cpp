#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>
#include <limits>

// allocates memory for the map
GaussMap::GaussMap(const std::string params){
    YAML::Node config = YAML::LoadFile(params);

    mapRel.height = config["MapHeight"].as<int>();
    mapRel.width = config["MapWidth"].as<int>();
    mapRel.res = config["MapResolution"].as<int>();

    useMin = config["UseMinValue"].as<bool>();
    minCutoff = config["MinGaussValue"].as<float>();
    if(!useMin)
        minCutoff = std::numeric_limits<float>::min();

    mapInfo.cols = mapRel.width * mapRel.res;
    mapInfo.rows = mapRel.height * mapRel.res;
    mapInfo.elementSize = sizeof(mapType_t);

    checkCudaError(cudaMalloc(&array, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));
    // allocate memory for the radar ids
    checkCudaError(cudaMalloc(&radarIds, sizeof(unsigned long long int) * mapInfo.rows * mapInfo.cols));

    radarDistri = (distInfo_t*)malloc(sizeof(struct DistributionInfo));
    radarDistri->stdDev = config["Radar"]["StdDev"].as<float>();
    radarDistri->mean = config["Radar"]["Mean"].as<float>();
    radarDistri->distCutoff = config["Radar"]["RadCutoff"].as<float>();

    // allocate this struct in shared memory so we don't have to copy
    // it to each kernel when it's needed
    checkCudaError(cudaMalloc(&mapInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&radarInfo_cuda, sizeof(struct Array_Info)));
    checkCudaError(cudaMalloc(&mapRel_cuda, sizeof(struct Array_Relationship)));
    checkCudaError(cudaMalloc(&radarDistri_c, sizeof(distInfo_t)));
    
    checkCudaError(cudaMemcpy(mapInfo_cuda, &mapInfo, sizeof(struct Array_Info), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(mapRel_cuda, &mapRel, sizeof(struct Array_Relationship), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(radarDistri_c, radarDistri, sizeof(distInfo_t), cudaMemcpyHostToDevice));

    // this is all done so we can check if it has been allocated later
    radarData = nullptr;
    reset();
}

GaussMap::~GaussMap(){
    safeCudaFree(array);
    safeCudaFree(mapInfo_cuda);
    safeCudaFree(mapRel_cuda);
    safeCudaFree(radarIds);

    if(radarData != nullptr){
        checkCudaError(cudaFree(radarData));
        safeCudaFree(radarInfo_cuda);
    }

    free(radarDistri);

    safeCudaFree(radarDistri_c);
}

void GaussMap::reset(){
    checkCudaError(cudaMemset(array, 0, mapInfo.cols * mapInfo.rows * mapInfo.elementSize));
    checkCudaError(cudaMemset(radarIds, 0, sizeof(unsigned long long int) * mapInfo.rows * mapInfo.cols));

    safeCudaFree(radarData);

    radarData = nullptr;
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

    radarPoints = buf1.shape[0];            // num points
    if(radarPoints == 0) return;            // do nothing if there are no points;
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
py::array_t<float> GaussMap::findMax(){
    std::pair<array_info,float*> maxima = calcMax();

    int rows = maxima.first.rows;

    py::buffer_info ret(
        maxima.second,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        {rows, (int)maxima.first.cols},
        {sizeof(float) * maxima.first.cols, sizeof(float) * 1}
    );
    return py::array_t<float>(ret);
}

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<std::string>())
        .def("reset", &GaussMap::reset)
        .def("addRadarData", &GaussMap::addRadarData)
        .def("asArray", &GaussMap::asArray, py::return_value_policy::take_ownership)
        .def("findMax", &GaussMap::findMax, py::return_value_policy::take_ownership);
}
