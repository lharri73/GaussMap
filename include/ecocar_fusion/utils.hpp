#include "types.hpp"

#define array_index_cpu(row,col,info) array_index_macro_cpu(row,col,info,__LINE__,__FILE__)

size_t array_index_macro_cpu(size_t row, size_t col, const array_info *info, int line, const char* file);
void print_arr(const float* arr, const array_info *info);
