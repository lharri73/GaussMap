#include "gaussMap.hpp"
#include "utils.hpp"
#include <iostream>

// allocates memory for the map
GaussMap::GaussMap(int Width, int Height, int Cell_res, double radarStdDev, double radarMean, double radarCutoff){
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

    radarDistri = (float*)calloc(3, sizeof(float));
    radarDistri[0] = (float)radarStdDev;
    radarDistri[1] = (float)radarMean;
    radarDistri[2] = (float)radarCutoff;

    // this is all done so we can check if it has been allocated later
    radarData = nullptr;
    arrayPrime = nullptr;
    arrayPrimePrime = nullptr;

    mapInfo_cuda = nullptr;
    mapRel_cuda = nullptr;
    radarInfo_cuda = nullptr;
    primeInfo_cuda = nullptr;
    primePrimeInfo_cuda = nullptr;

    allClean = false;
}

GaussMap::~GaussMap(){
    if(!allClean)
        cleanup();
}

void GaussMap::cleanup(){
    // performs the cleanup steps. Frees memory
    if(!allClean){
        checkCudaError(cudaFree(array));
        if(mapInfo_cuda != nullptr)
            checkCudaError(cudaFree(mapInfo_cuda));
        if(mapRel_cuda != nullptr)
            checkCudaError(cudaFree(mapRel_cuda));
        if(arrayPrime != nullptr)
            checkCudaError(cudaFree(arrayPrime));
        if(arrayPrimePrime != nullptr)
            checkCudaError(cudaFree(arrayPrimePrime));
        

        if(radarData != nullptr){
            checkCudaError(cudaFree(radarData));
            if(radarInfo_cuda != nullptr)
                checkCudaError(cudaFree(radarInfo_cuda));
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
    mapType_t* retArray = new mapType_t[mapInfo.cols * mapInfo.rows];

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

// calculate the first and second derivative of the heatmap and
// return as a tuple (first, second) in a numpy array
std::vector<py::array_t<float> > GaussMap::derivative(){
    calcDerivative();
    std::vector<py::array_t<float> > ret;
    float* firstDeriv = new float[primeInfo.cols * primeInfo.rows];

    checkCudaError(cudaMemcpy(firstDeriv, arrayPrime, primeInfo.cols * primeInfo.rows * primeInfo.elementSize, cudaMemcpyDeviceToHost));

    py::buffer_info a(
        firstDeriv,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        {primeInfo.rows, primeInfo.cols},
        {sizeof(float) * primeInfo.cols, sizeof(float) * 1});
    
    ret.push_back(py::array_t<float>(a));

    float* secondDeriv = new float[primePrimeInfo.cols * primePrimeInfo.rows];
    checkCudaError(cudaMemcpy(secondDeriv, arrayPrimePrime, primePrimeInfo.cols * primePrimeInfo.rows * primeInfo.elementSize, cudaMemcpyDeviceToHost));
    
    py::buffer_info b(
        secondDeriv,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        {primePrimeInfo.rows, primePrimeInfo.cols},
        {sizeof(float) * primePrimeInfo.cols, sizeof(float) * 1});
    ret.push_back(py::array_t<float>(b));

    return ret;
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

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int,double,double,double>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData)
        .def("asArray", &GaussMap::asArray, py::return_value_policy::take_ownership)
        .def("derivative", &GaussMap::derivative, py::return_value_policy::take_ownership)
        .def("findMax", &GaussMap::findMax, py::return_value_policy::take_ownership);
}
