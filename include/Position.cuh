#include <cuda_runtime.h>

// simple class used to hold the position information passed between cuda functions
class Position{
    public:
        __device__ Position(float x, float y);
        __device__ Position();
        __device__ void recalc();
        
        float x;
        float y;
        float radius;

};