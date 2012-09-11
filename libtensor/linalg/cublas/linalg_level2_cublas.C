#include <cublas_v2.h>
#include "linalg_level2_cublas.h"

namespace libtensor {


const char *linalg_level2_cublas::k_clazz = "cublas";


void linalg_level2_cublas::i_ip_p_x(
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cublasHandle_t h;
    const double one = 1.0;
    start_timer("dgemv");
    cublasDgemv(h, CUBLAS_OP_T, np, ni, &d, a, sia, b, spb, &one, c, sic);
    stop_timer("dgemv");
}


void linalg_level2_cublas::i_pi_p_x(
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cublasHandle_t h;
    const double one = 1.0;
    start_timer("dgemv");
    cublasDgemv(h, CUBLAS_OP_N, ni, np, &d, a, spa, b, spb, &one, c, sic);
    stop_timer("dgemv");
}


void linalg_level2_cublas::ij_i_j_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    cublasHandle_t h;
    start_timer("dger");
    cublasDger(h, nj, ni, &d, b, sjb, a, sia, c, sic);
    stop_timer("dger");
}


void linalg_level2_cublas::ij_ji_x(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    cublasHandle_t h;
    start_timer("dcopy+dscal");
    for(size_t i = 0; i < ni; i++) {
        cublasDcopy(h, nj, a + i, sja, c + i * sic, 1);
        cublasDscal(h, nj, &b, c + i * sic, 1);
    }
    stop_timer("dcopy+dscal");
}


} // namespace libtensor
