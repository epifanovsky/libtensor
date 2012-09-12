#include <cublas_v2.h>
#include "linalg_level1_cublas.h"

namespace libtensor {


const char *linalg_level1_cublas::k_clazz = "cublas";


void linalg_level1_cublas::i_x(
    size_t ni,
    double a,
    double *c, size_t sic) {

    cublasHandle_t h;
    start_timer("dscal");
    cublasStatus_t ec = cublasDscal(h, ni, &a, c, sic);
    stop_timer("dscal");
}


double linalg_level1_cublas::x_p_p(
    cublasHandle_t h,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    start_timer("ddot");
    double d = 0.0;
    cublasStatus_t ec = cublasDdot(h, np, a, spa, b, spb, &d);
    stop_timer("ddot");
    return d;
}


void linalg_level1_cublas::i_i_x(
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    cublasHandle_t h;
    start_timer("daxpy");
    cublasStatus_t ec = cublasDaxpy(h, ni, &b, a, sia, c, sic);
    stop_timer("daxpy");
}


} // namespace libtensor
