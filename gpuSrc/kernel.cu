#include <sstream>
#include <iostream>
#include <cuda_runtime.h>


// __global__ size_t array_index
// (int x, int y, ){

// }


void run_kernel
(double *vec, double scalar, int num_elements)
{
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
  
  kernel<<<dimGrid, dimBlock>>>
    (vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
