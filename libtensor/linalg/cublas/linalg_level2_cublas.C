#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "linalg_level2_cublas.h"

namespace libtensor {


const char *linalg_level2_cublas::k_clazz = "cublas";


void linalg_level2_cublas::copy_ij_ji_x(
    cublasHandle_t h,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    cublasStatus_t ec;
    start_timer("dcopy+dscal");
    for(size_t i = 0; i < ni; i++) {
        ec = cublasDcopy(h, nj, a + i, sja, c + i * sic, 1);
        ec = cublasDscal(h, nj, &b, c + i * sic, 1);
    }
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dcopy+dscal");
}


void linalg_level2_cublas::mul2_i_ip_p_x(
    cublasHandle_t h,
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemv");
    cublasStatus_t ec = cublasDgemv(h, CUBLAS_OP_T, np, ni, &d, a, sia, b, spb,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemv");
}


void linalg_level2_cublas::mul2_i_pi_p_x(
    cublasHandle_t h,
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemv");
    cublasStatus_t ec = cublasDgemv(h, CUBLAS_OP_N, ni, np, &d, a, spa, b, spb,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemv");
}


void linalg_level2_cublas::mul2_ij_i_j_x(
    cublasHandle_t h,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dger");
    cublasStatus_t ec = cublasDger(h, nj, ni, &d, b, sjb, a, sia, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dger");
}


} // namespace libtensor
