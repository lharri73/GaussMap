cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(gaussMap LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11)
set(PYBIND11_CPP_STANDARD -std=c++11)

include_directories(
    include 
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

## CUDA library
add_library(gaussMap_cu STATIC
    src/kernel.cu
    src/derivative.cu
)
set_target_properties(gaussMap_cu PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
)
set_property(TARGET gaussMap_cu PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72)
set_property(TARGET gaussMap_cu PROPERTY CXX_STANDARD 11)


## Link to cuda
target_link_libraries(gaussMap_cu PRIVATE
    CUDA::cudart
)

## C++ library
add_library(gaussMap MODULE
    src/gaussMap.cpp
    src/utils.cpp
)

set_target_properties(gaussMap PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
set_property(TARGET gaussMap PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72)
set_property(TARGET gaussMap PROPERTY CXX_STANDARD 11)
## link together and with pybind11
target_link_libraries(gaussMap PRIVATE gaussMap_cu)
target_link_libraries(gaussMap PRIVATE pybind11::module)

