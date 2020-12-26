#include <cstdio>
#include "header.hpp"

int main(int argc, char** argv){
    printf("here\n");
    return 0;
}

const int* gaussMap::Array() const{
    return array;
}

size_t gaussMap::Width() const{
    return width;
}

size_t gaussMap::Height() const{
    return height;
}
