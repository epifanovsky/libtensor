#include "cblas_h.h"
#include "linalg_cblas_level2.h"

namespace libtensor {

template<>
const char *linalg_cblas_level2<double>::k_clazz = "cblas";


template<>
void linalg_cblas_level2<double>::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            cblas_dcopy(nj, a + i, sja, c1, 1);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            cblas_dcopy(ni, a1, 1, c + j, sic);
        }
    }
}


template<>
void linalg_cblas_level2<double>::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b,
        spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level2<double>::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b,
        spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level2<double>::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
}


template<>
const char *linalg_cblas_level2<float>::k_clazz = "cblas";


template<>
void linalg_cblas_level2<float>::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sja,
    float *c, size_t sic) {

    if(ni < nj) {
        float *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            cblas_scopy(nj, a + i, sja, c1, 1);
        }
    } else {
        const float *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            cblas_scopy(ni, a1, 1, c + j, sic);
        }
    }
}


template<>
void linalg_cblas_level2<float>::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const float *a, size_t sia,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b,
        spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level2<float>::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const float *a, size_t spa,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    cblas_sgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b,
        spb, 1.0, c, sic);
}


template<>
void linalg_cblas_level2<float>::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sia,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    cblas_sger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
}


} // namespace libtensor
