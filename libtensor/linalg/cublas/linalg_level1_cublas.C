#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <libtensor/cuda/cuda_error.h>
#include "linalg_level1_cublas.h"

namespace libtensor {


const char *linalg_level1_cublas::k_clazz = "cublas";


void linalg_level1_cublas::copy_i_i(
    cublasHandle_t h,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    start_timer("copy_i_i");
    stop_timer("copy_i_i");
}


void linalg_level1_cublas::mul1_i_x(
    cublasHandle_t h,
    size_t ni,
    double a,
    double *c, size_t sic) {

    start_timer("dscal");
    cublasStatus_t ec = cublasDscal(h, ni, &a, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);

    stop_timer("dscal");
}


double linalg_level1_cublas::mul2_x_p_p(
    cublasHandle_t h,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    static const char method[] = "mul2_x_p_p()";

    start_timer("ddot");
    double d = 0.0;
    cublasStatus_t ec = cublasDdot(h, np, a, spa, b, spb, &d);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        stop_timer("ddot");
        throw cuda_error(g_ns, k_clazz, method, __FILE__, __LINE__, "ec");
    }
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        stop_timer("ddot");
        throw cuda_error(g_ns, k_clazz, method, __FILE__, __LINE__, "ec");
    }
    cudaStreamSynchronize(stream);
    stop_timer("ddot");
    return d;
}


void linalg_level1_cublas::mul2_i_i_x(
    cublasHandle_t h,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    start_timer("daxpy");
    cublasStatus_t ec = cublasDaxpy(h, ni, &b, a, sia, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("daxpy");
}


void linalg_level1_cublas::mul2_i_i_i_x(
    cublasHandle_t h,
    size_t ni,
    const double *a, size_t sia,
    const double *b, size_t sib,
    double *c, size_t sic,
    double d) {

    start_timer("mul2_i_i_i_x");
    stop_timer("mul2_i_i_i_x");
}


} // namespace libtensor
