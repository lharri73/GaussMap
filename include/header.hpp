#include <pybind11/pybind11.h>
#include <cstdlib>

class GaussMap{
    private:
        cudaArray_t array;
        int height, width;
        int vcells, hcells;
        bool allClean;
    public:
        GaussMap(int width, int height, int vcells, int hcells);
        ~GaussMap();
        void cleanup();

};

// void run_kernel(double *vec, double scalar, int num_elements);
// void multiply_with_scalar(pybind11::array_t<double> vec, double scalar);

namespace py = pybind11;
PYBIND11_MODULE(gaussMap, m){
    // m.def("multiply_with_scalar", multiply_with_scalar);
    py::class_<GaussMap>(m,"GaussMap")
        .def(py::init<int,int,int,int>())
        .def("cleanup", &GaussMap::cleanup);
}
