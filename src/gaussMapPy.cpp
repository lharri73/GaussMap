/*
Implementation of the GaussMap class that requires python bindings.
Every function not in this file should not require python
*/
#include "gaussMap.hpp"
#include "cudaUtils.hpp"

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMap::addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array){
    if(radarData != nullptr)
        throw std::runtime_error("addRadarData can only be called once after calling reset()");
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
    if(radarFeatures != 6){
        throw std::runtime_error("Got invalid shape of Radar Data. should be Nx6");
    }

    radarInfo.elementSize = sizeof(RadarData_t);
    radarInfo.cols = radarFeatures;
    radarInfo.rows = radarPoints;
    printf("radPoints: %zu\n", radarPoints);

    // allocate and copy the array to the GPU so we can run a kernel on it
    safeCudaMalloc(&radarData, sizeof(RadarData_t) * radarPoints * radarFeatures);

    safeCudaMemcpy2Device(radarData, data, sizeof(RadarData_t) * radarPoints * radarFeatures);

    calcRadarMap();     // setup for the CUDA kernel. in GPU code
}


// returns numpy array to python of gaussMap
py::array_t<mapType_t> GaussMap::asArray(){
    mapType_t* retArray = new mapType_t[mapInfo.cols * mapInfo.rows];

    safeCudaMemcpy2Host(retArray, array, mapInfo.size());

    py::buffer_info a(
        retArray, 
        sizeof(mapType_t), 
        py::format_descriptor<mapType_t>::format(), 
        2, 
        {mapInfo.rows, mapInfo.cols},
        {sizeof(mapType_t) * mapInfo.cols, sizeof(mapType_t) * 1});
    
    return py::array_t<mapType_t>(a);
}

void GaussMap::addCameraData(py::array_t<float, py::array::c_style | py::array::forcecast> array){
    py::buffer_info buf1 = array.request();
    float* data;
    data = static_cast<float*>(buf1.ptr);
    if(buf1.itemsize != sizeof(float)){
        throw std::runtime_error("Invalid datatype passed with camera data. Expected float32");
    }

    camInfo.cols = buf1.shape[1];
    camInfo.rows = buf1.shape[0];
    camInfo.elementSize = buf1.itemsize;

    safeCudaMalloc(&camData, camInfo.elementSize * camInfo.cols * camInfo.rows);
    safeCudaMemcpy2Device(camData, data, camInfo.elementSize * camInfo.cols * camInfo.rows);

    safeCudaMemcpy2Device(camInfo_cuda, &camInfo, sizeof(array_info));

}

py::array_t<float> GaussMap::associate(){

    // return: [x,y,vx,vy,class]

    if(radarData == nullptr || camData == nullptr)
        throw std::runtime_error("Radar and Camera data must be added before association!");

    std::pair<array_info,float*> associated = associatePair();

    py::buffer_info ret(
        associated.second,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        {(int)associated.first.rows, (int)associated.first.cols},
        {sizeof(float) * associated.first.cols, sizeof(float) * 1}
    );
    return py::array_t<float>(ret);
}

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<std::string>())
        .def("reset", &GaussMap::reset)
        .def("addRadarData", &GaussMap::addRadarData)
        .def("addCameraData", &GaussMap::addCameraData)
        .def("asArray", &GaussMap::asArray, py::return_value_policy::take_ownership)
        .def("associate", &GaussMap::associate, py::return_value_policy::take_ownership);
}
