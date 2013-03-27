#ifndef LIBTENSOR_CUDA_UTILS_H
#define LIBTENSOR_CUDA_UTILS_H

#include <libtensor/cuda/cuda_error.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace libtensor {


/** \brief Set of utils to be used with CUDA calls

    \ingroup libtensor_cuda
 **/
class cuda_utils {
public:

	static void handle_error( cudaError_t err, const char *ns, const char *clazz,
			                   const char *method, const char *file,  int line ) {

	   if (  err != cudaSuccess) {
		   throw cuda_error(ns, clazz, method, file, line, cudaGetErrorString( err ));
	   }
	}

	static void handle_kernel_error( const char *ns, const char *clazz,
				                   const char *method, const char *file,  int line ) {

	   cudaError_t err = cudaGetLastError();
	   if (  err != cudaSuccess) {
		   throw cuda_error(ns, clazz, method, file, line, cudaGetErrorString( err ));
	   }
	}

	static void handle_cublas_error( cublasStatus_t err, const char *ns, const char *clazz,
				                   const char *method, const char *file,  int line ) {

	   if (  err != CUBLAS_STATUS_SUCCESS) {
		   throw cuda_error(ns, clazz, method, file, line, cublas_get_error_string( err ));
	   }
	}

	static const char *cublas_get_error_string(cublasStatus_t error)
	{
	    switch (error)
	    {
	        case CUBLAS_STATUS_SUCCESS:
	            return "CUBLAS_STATUS_SUCCESS";

	        case CUBLAS_STATUS_NOT_INITIALIZED:
	            return "CUBLAS_STATUS_NOT_INITIALIZED";

	        case CUBLAS_STATUS_ALLOC_FAILED:
	            return "CUBLAS_STATUS_ALLOC_FAILED";

	        case CUBLAS_STATUS_INVALID_VALUE:
	            return "CUBLAS_STATUS_INVALID_VALUE";

	        case CUBLAS_STATUS_ARCH_MISMATCH:
	            return "CUBLAS_STATUS_ARCH_MISMATCH";

	        case CUBLAS_STATUS_MAPPING_ERROR:
	            return "CUBLAS_STATUS_MAPPING_ERROR";

	        case CUBLAS_STATUS_EXECUTION_FAILED:
	            return "CUBLAS_STATUS_EXECUTION_FAILED";

	        case CUBLAS_STATUS_INTERNAL_ERROR:
	            return "CUBLAS_STATUS_INTERNAL_ERROR";
	    }
	    return "<unknown>";
	}
};

} // namespace libtensor

#endif // LIBTENSOR_CUDA_UTILS_H
