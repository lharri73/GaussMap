/* All cuda functions return a cudaError_t, this function makes sure that it is
 * success. If not, it throws a runtime error. This is usefull for debugging 
 * rather than getting a segmentation fault */
template <typename T>
void safeCudaMalloc(T **ptr, size_t size){
    cudaError_t error = cudaMalloc(ptr, size);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMalloc\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Device(T *dst, const T *src, size_t size){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Device\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemcpy2Host(T *dst, const T *src, size_t size){
    cudaError_t error = cudaMemcpy(dst,src, size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemcpy2Host\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void safeCudaMemset(T* ptr, int value, size_t size){
    cudaError_t error = cudaMemset(ptr, value, size);
    if(error != cudaSuccess){
        std::stringstream ss;
        ss << "gaussMap:: Internal error during cudaMemset\n";
        ss << "\tCUDA: " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}