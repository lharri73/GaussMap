#include "Position.cuh"

// implementation of the Position class.
__device__ 
Position::Position(float X, float Y) : x(X), y(Y){
    radius = hypotf(x,y);
}

__device__
Position::Position(){

}

__device__
void Position::recalc(){
    radius = hypotf(x,y);
}