#include <mkl.h>
#include "linalg_base_level3_mkl.h"

namespace libtensor {


const char *linalg_mkl_level3::k_clazz = "mkl";


void linalg_mkl_level3::ij_ip_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np,
        d, a, sia, b, sjb, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_mkl_level3::ij_ip_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj,
        np, d, a, sia, b, spb, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_mkl_level3::ij_pi_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np,
        d, a, spa, b, sjb, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_mkl_level3::ij_pi_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np,
        d, a, spa, b, spb, 1.0, c, sic);
    stop_timer("dgemm");
}


} // namespace libtensor

