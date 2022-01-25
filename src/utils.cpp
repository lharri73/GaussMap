#ifndef NUSCENES
#include "ecocar_fusion/cudaUtils.hpp"
#include "ecocar_fusion/utils.hpp"
#else
#include "cudaUtils.hpp"
#include "utils.hpp"
#endif

void checkCudaError(cudaError_t error){
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap internal error...\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

void safeCudaFree_macro(void *ptr, int line, const char* file){
    if(ptr != nullptr){
        cudaError_t error = cudaFree(ptr);
        if(error != cudaSuccess){
            std::stringstream ss;
            ss << "gaussMap:: Internal error during cudaFree at " << file << ":" << line << '\n';
            ss << "\tCUDA: " << cudaGetErrorString(error) << '\n';
            throw std::runtime_error(ss.str());
        }
    }
}

size_t array_info::size(){
    return rows * cols * elementSize;
}

size_t array_index_macro_cpu(size_t row, size_t col, const array_info *info, int line, const char* file){
    if(row > info->rows){
        fprintf(stderr, "Index out of bounds at %s::%d (info->row < row)\n", file, line);
        exit(1);
    }
    if(col > info->cols){
        fprintf(stderr, "Index out of bounds at %s::%d (info->col < col)\n", file, line);
        exit(1);
    }
    return (row * info->cols) + col;
}


void print_arr(const float* arr, const array_info *info){
    printf("+++\n");
    for(size_t i = 0; i < info->rows; i++){
        printf("[");
        for(size_t j = 0; j < info->cols; j++){
            printf("%6.2f,", arr[array_index_cpu(i, j, info)]);
        }
        printf("]\n");
    }
    printf("---\n");
}
