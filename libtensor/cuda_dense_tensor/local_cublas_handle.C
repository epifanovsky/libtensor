#include <iostream>
#include "local_cublas_handle.h"

namespace libtensor {


local_cublas_handle::local_cublas_handle() {

    cublasStatus_t ec = cublasCreate(&m_handle);
    if(ec != cudaSuccess) {
        std::cout << "Unable to create cuBLAS handle" << std::endl;
    }
}


local_cublas_handle::~local_cublas_handle() {

    cublasDestroy(m_handle);
}


} // namespace libtensor

