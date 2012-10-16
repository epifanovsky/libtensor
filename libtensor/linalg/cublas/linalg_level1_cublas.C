#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "linalg_level1_cublas.h"

namespace libtensor {


const char *linalg_level1_cublas::k_clazz = "cublas";


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

    start_timer("ddot");
    double d = 0.0;
    cublasStatus_t ec = cublasDdot(h, np, a, spa, b, spb, &d);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
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


} // namespace libtensor
