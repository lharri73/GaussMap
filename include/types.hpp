#pragma once
#include <cstdint>
#include <cstdlib>

typedef float RadarData_t;
typedef float mapType_t;

typedef struct Array_Info{
    size_t rows;            // total number of rows
    size_t cols;            // total number of columns
    size_t elementSize;     // size of a single element in bytes
    size_t size();          // utils.cpp
} array_info;

typedef struct Array_Relationship{
    size_t width;           // meters
    size_t height;          // meters
    size_t res;             // resolution (cells per linear meter)
} array_rel;

typedef struct MaxVal{
    uint8_t isMax;
    uint8_t classVal;
    int16_t radars[49];
} maxVal_t;

typedef struct DistributionInfo{
    float mean;
    float stdDev;
    float distCutoff;
} distInfo_t;

typedef struct RadarIds{
    int16_t radarId;       // uint32 so it's aligned to double word
    int16_t garbage;
    float probability;     // value of pdf
} radarId_t; 
