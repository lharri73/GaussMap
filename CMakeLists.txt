cmake_minimum_required(VERSION 3.8)
project(gaussMap)

enable_language(CUDA)

include_directories(include)
add_executable(gaussMap 
    src/main.cpp
)
