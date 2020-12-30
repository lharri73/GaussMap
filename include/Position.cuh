#include <cuda_runtime.h>

class Position{
    public:
        __device__ Position(float x, float y);
        __device__ Position();
        __device__ void recalc();
        
        float x;
        float y;
        float radius;

};