#include <iostream>
#include "local_cublas_handle.h"
#include <libtensor/cuda/cuda_utils.h>

namespace libtensor {

const char *local_cublas_handle::k_clazz = "local_cublas_handle";

local_cublas_handle::local_cublas_handle() {
	 static const char method[] = "local_cublas_handle()";

    cublasStatus_t ec = cublasCreate(&m_handle);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Unable to create cuBLAS handle" << std::endl;
        throw cuda_error(g_ns, k_clazz, method, __FILE__, __LINE__, cuda_utils::cublas_get_error_string(ec));
    }
}


local_cublas_handle::~local_cublas_handle() {

    cublasDestroy(m_handle);
}


} // namespace libtensor

