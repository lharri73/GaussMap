// #include <iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "header.hpp"

GaussMap::GaussMap(int Width, int Height, int Vcells, int Hcells) : 
    height{Height}, width{Width}, vcells{Vcells}, hcells{Hcells}{
    // allocate memory for the array
    cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKind::cudaChannelFormatKindSigned;
    desc.x = 8; // use 8 bits for the x fields
    // dont use the y,z,w fields
    desc.y = 0;
    desc.z = 0;
    desc.w = 0;

    cudaError_t error = cudaMallocArray(&array, &desc, width, height);
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }
    allClean = false;
}

GaussMap::~GaussMap(){
    // there isn't a nice way to call destructors from 
    // python, so we do it this way. 
    if(!allClean)
        cleanup();
}

void GaussMap::cleanup(){
    cudaError_t error = cudaFreeArray(array);
    if(error != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(error));
    }
    allClean = true;
}