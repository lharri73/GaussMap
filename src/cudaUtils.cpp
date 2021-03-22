/* All cuda functions return a cudaError_t, this function makes sure that it is
 * success. If not, it throws a runtime error. This is usefull for debugging 
 * rather than getting a segmentation fault */
template <typename T>
void safeCudaMalloc_macro(T **ptr, size_t size, int line, const char* file){
    cudaError_t error = cudaMalloc(ptr, size);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMalloc at " << file << ":" << line << '\n';
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Device_macro(T *dst, const T *src, size_t size, int line, const char* file){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Device at " << file << ":" << line << '\n';
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Host_macro(T *dst, const T *src, size_t size, int line, const char* file){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Host at " << file << ":" << line << '\n';
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemset_macro(T* ptr, int value, size_t size, int line, const char* file){
    cudaError_t error = cudaMemset(ptr, value, size);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemset at " << file << ":" << line << '\n';
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}