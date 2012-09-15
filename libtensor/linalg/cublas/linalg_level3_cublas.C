#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "linalg_level3_cublas.h"

namespace libtensor {


const char *linalg_level3_cublas::k_clazz = "cublas";


void linalg_level3_cublas::ij_ip_jp_x(
    cublasHandle_t h,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemm");
    cublasStatus_t ec = cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nj, ni, np, &d, b, sjb, a, sia,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemm");
}


void linalg_level3_cublas::ij_ip_pj_x(
    cublasHandle_t h,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemm");
    cublasStatus_t ec = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, nj, ni, np, &d, b, spb, a, sia,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemm");
}


void linalg_level3_cublas::ij_pi_jp_x(
    cublasHandle_t h,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemm");
    cublasStatus_t ec = cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, nj, ni, np, &d, b, sjb, a, spa,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemm");
}


void linalg_level3_cublas::ij_pi_pj_x(
    cublasHandle_t h,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    const double one = 1.0;
    start_timer("dgemm");
    cublasStatus_t ec = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, nj, ni, np, &d, b, spb, a, spa,
        &one, c, sic);
    cudaStream_t stream;
    ec = cublasGetStream(h, &stream);
    cudaStreamSynchronize(stream);
    stop_timer("dgemm");
}


} // namespace libtensor
