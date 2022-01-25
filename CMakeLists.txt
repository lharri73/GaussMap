cmake_minimum_required(VERSION 3.17)
project(gaussMap LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11)
find_package(yaml-cpp REQUIRED)
add_definitions(-DNUSCENES)


include_directories(
    include/ecocar_fusion
    src
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

pybind11_add_module(gaussMap THIN_LTO NO_EXTRAS
    src/gaussMap.cpp
    src/gaussMapPy.cpp
    src/utils.cpp
    src/gpu/associate.cu
    src/gpu/calcMax.cu
    src/gpu/gaussMapGPU.cu
    src/gpu/radarPoint.cu
    src/gpu/reset.cu
)

## Link to cuda
target_link_libraries(gaussMap
    #CUDA::cudart
    ${YAML_CPP_LIBRARIES}
)
set_property(TARGET gaussMap PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET gaussMap PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72)
set_property(TARGET gaussMap PROPERTY CXX_STANDARD 14)

set(PYBIND11_CPP_STANDARD -std=c++14)

