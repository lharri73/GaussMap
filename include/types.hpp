#pragma once
#include <cstdint>
#include <cstdlib>

typedef float RadarData_t;
typedef float mapType_t;

typedef struct Array_Info{
    size_t rows;            // total number of rows
    size_t cols;            // total number of columns
    size_t elementSize;     // size of a single element in bytes
} array_info;

typedef struct Array_Relationship{
    size_t width;           // meters
    size_t height;          // meters
    size_t res;             // resolution (cells per linear meter)
} array_rel;

typedef struct CamVal{
    uint32_t classVal;      // uint16 so it's aligned to dopuble word (32 bits)
    float probability;      // value of pdf
} camVal_t;