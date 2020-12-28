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

        size_t numPoints;
        size_t radarFeatures;
    public:
        GaussMap(int width, int height, int vcells, int hcells);
        ~GaussMap();
        void cleanup();
        void addRadarData(py::array_t<RadarData_t, py::array::c_style | py::array::forcecast> array);

};

// void run_kernel(double *vec, double scalar, int num_elements);
// void multiply_with_scalar(pybind11::array_t<double> vec, double scalar);

PYBIND11_MODULE(gaussMap, m){
    // m.def("multiply_with_scalar", multiply_with_scalar);
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int,int>())
        .def("cleanup", &GaussMap::cleanup)
        .def("addRadarData", &GaussMap::addRadarData);
}
