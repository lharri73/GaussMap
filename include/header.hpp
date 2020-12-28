// #include <cstdlib>

// class gaussMap{
//     private:
//         int* array;
//         size_t width;
//         size_t height;

//     public:
//         const int* Array() const;
//         size_t Width() const;
//         size_t Height() const;
// };
#include <pybind11/pybind11.h>
void run_kernel(double *vec, double scalar, int num_elements);
void multiply_with_scalar(pybind11::array_t<double> vec, double scalar);

PYBIND11_MODULE(gaussMap, m)
{
  m.def("multiply_with_scalar", multiply_with_scalar);
}
