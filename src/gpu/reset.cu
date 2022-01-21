#ifndef NUSCENES
#include "ecocar_fusion/gaussMap.cuh"
#else
#include "gaussMap.cuh"
#endif

/* Functions like memset, but since cudaMemset takes an integer, 
 * this is necessary. This assigns an unsigned long long int to the
 * memory address of every element in the array. Since it's a kernel,
 * each thread does one operation */
__global__
void setRadarIdsKernel(radarId_t *array){
    array[blockIdx.x].radarId = -1;
    array[blockIdx.x].garbage = 0;
    array[blockIdx.x].probability = 0.0;
}

//-----------------------------------------------------------------------------
