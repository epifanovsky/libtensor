#ifndef LIBTENSOR_CUDA_ALLOCATOR_IMPL_H
#define LIBTENSOR_CUDA_ALLOCATOR_IMPL_H

#include <cuda_runtime_api.h>
#include "../cuda_allocator.h"
#include "../cuda_exception.h"

namespace libtensor {


template<typename T>
typename cuda_allocator<T>::pointer_type
cuda_allocator<T>::allocate(size_t sz) {

    static const char *method = "allocate(size_t)";

    pointer_type dp;
    cudaError_t ec = cudaMalloc((void**)&dp.p, sizeof(T) * sz);
    if(ec != cudaSuccess) {
        throw cuda_exception(k_clazz, method, __FILE__, __LINE__,
            cudaGetErrorString(ec));
    }
    return dp;
}


template<typename T>
void cuda_allocator<T>::deallocate(pointer_type dp) {

    static const char *method = "deallocate(pointer_type)";

    cudaError_t ec = cudaFree(dp.p);
    if(ec != cudaSuccess) {
        throw cuda_exception(k_clazz, method, __FILE__, __LINE__,
            cudaGetErrorString(ec));
    }
}


template<typename T>
void cuda_allocator<T>::copy_to_device(pointer_type dp, const T *hp, size_t sz) {

    static const char *method = "copy_to_device(pointer_type, const T*, size_t)";

    cudaError_t ec = cudaMemcpy(dp.p, hp, sizeof(T) * sz, cudaMemcpyHostToDevice);
    if(ec != cudaSuccess) {
        throw cuda_exception(k_clazz, method, __FILE__, __LINE__,
            cudaGetErrorString(ec));
    }
}


template<typename T>
void cuda_allocator<T>::copy_to_host(T * hp, pointer_type dp, size_t sz) {

    static const char *method = "copy_to_host(T*, pointer_type, size_t)";

    cudaError_t ec = cudaMemcpy(hp, dp.p, sizeof(T) * sz, cudaMemcpyDeviceToHost);
    if(ec != cudaSuccess) {
        throw cuda_exception(k_clazz, method, __FILE__, __LINE__,
            cudaGetErrorString(ec));
    }
}


template<typename T>
const char *cuda_allocator<T>::k_clazz = "cuda_allocator<T>";


template<typename T>
const typename cuda_allocator<T>::pointer_type
cuda_allocator<T>::invalid_pointer;


} // namespace libtensor

#endif // LIBTENSOR_CUDA_ALLOCATOR_IMPL_H
