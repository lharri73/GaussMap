/*
Implementation of the GaussMap class that requires python bindings.
Every function not in this file should not require python
*/
#include "gaussMapPy.hpp"
#include "cudaUtils.hpp"

GaussMapPy::GaussMapPy(const std::string params){
    ROS_INFO_STREAM("[ecocar_fusion_node] Initializing GaussMap...");

    int mapWidth;               // Gauss map width
    int mapHeight;              // Gauss map height
    int mapResolution;          // Gauss map resolution (meter per pixel?)
    radarDistri = (distInfo_t*)malloc(sizeof(struct DistributionInfo));

    YAML::Node config = YAML::LoadFile(params);

    mapHeight = config["MapHeight"].as<int>();
    mapWidth = config["MapWidth"].as<int>();
    mapResolution = config["MapResolution"].as<int>();
    adjustFactor = config["adjustFactor"].as<float>();

    useMin = config["UseMinValue"].as<bool>();
    minCutoff = config["MinGaussValue"].as<float>();

    radarDistri = (distInfo_t*)malloc(sizeof(struct DistributionInfo));
    radarDistri->stdDev = config["Radar"]["StdDev"].as<float>();
    radarDistri->mean = config["Radar"]["Mean"].as<float>();
    radarDistri->distCutoff = config["Radar"]["RadCutoff"].as<float>();
    GaussMap::init(mapHeight, mapWidth, mapResolution,useMin);
}

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMapPy::addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array){
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
    radarFeatures = buf1.shape[1];

    radarInfo.elementSize = sizeof(RadarData_t);
    radarInfo.cols = radarFeatures;
    radarInfo.rows = radarPoints;

    // allocate and copy the array to the GPU so we can run a kernel on it
    safeCudaMalloc(&radarData, sizeof(RadarData_t) * radarPoints * radarFeatures);
    // for(size_t i = 0; i < radarInfo.size() / sizeof(RadarData_t); i++){
    //     if(i % radarInfo.cols == 0)
    //         putchar('\n');
    //     printf("%.2f ", data[i]);
    // }
    // printf("\n------------------------\n\n");

    safeCudaMemcpy2Device(radarData, data, sizeof(RadarData_t) * radarPoints * radarFeatures);
    
    calcRadarMap();     // setup for the CUDA kernel. in GPU code
}


// returns numpy array to python of gaussMap
py::array_t<mapType_t> GaussMapPy::asArray(){
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

void GaussMapPy::addCameraData(py::array_t<float, py::array::c_style | py::array::forcecast> array){
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

py::array_t<float> GaussMapPy::associate(){

    // return: [x,y,vx,vy,class]

    if(radarData == nullptr || camData == nullptr)
        throw std::runtime_error("Radar and Camera data must be added before association!");

    std::pair<array_info,float*> associated = associatePair();

    py::buffer_info ret(
        associated.second,
        sizeof(float),
        py::format_descriptor<float>::format(),
        2,
        {associated.first.rows, associated.first.cols},
        {sizeof(float) * associated.first.cols, sizeof(float) * 1}
    );
    return py::array_t<float>(ret);
}

PYBIND11_MODULE(gaussMap, m){
    py::class_<GaussMapPy>(m,"GaussMap")
        .def(py::init<std::string>())
        .def("reset", &GaussMapPy::reset)
        .def("addRadarData", &GaussMapPy::addRadarData)
        .def("addCameraData", &GaussMapPy::addCameraData)
        .def("asArray", &GaussMapPy::asArray, py::return_value_policy::take_ownership)
        .def("associate", &GaussMapPy::associate, py::return_value_policy::take_ownership);
}
