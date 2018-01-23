#include "cblas_h.h"
#include "linalg_cblas_level3.h"

namespace libtensor {

template<>
const char *linalg_cblas_level3<double>::k_clazz = "cblas";


template<>
void linalg_cblas_level3<double>::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np,
        d, a, sia, b, sjb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<double>::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj,
        np, d, a, sia, b, spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<double>::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np,
        d, a, spa, b, sjb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<double>::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np,
        d, a, spa, b, spb, 1.0, c, sic);
}


template<>
const char *linalg_cblas_level3<float>::k_clazz = "cblas";


template<>
void linalg_cblas_level3<float>::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t sia,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np,
        d, a, sia, b, sjb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<float>::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t sia,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj,
        np, d, a, sia, b, spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<float>::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t spa,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np,
        d, a, spa, b, sjb, 1.0, c, sic);
}


template<>
void linalg_cblas_level3<float>::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t spa,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np,
        d, a, spa, b, spb, 1.0, c, sic);
}



} // namespace libtensor
