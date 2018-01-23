#include "cblas_h.h"
#include "linalg_cblas_level1.h"
#include <iostream>

namespace libtensor {


template<>
const char *linalg_cblas_level1<double>::k_clazz = "cblas";

template<>
void linalg_cblas_level1<double>::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    cblas_daxpy(ni, d * ka, a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
}


template<>
void linalg_cblas_level1<double>::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    cblas_dcopy(ni, a, sia, c, sic);
}


template<>
void linalg_cblas_level1<double>::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    cblas_dscal(ni, a, c, sic);
}


template<>
double linalg_cblas_level1<double>::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    return cblas_ddot(np, a, spa, b, spb);
}


template<>
void linalg_cblas_level1<double>::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    cblas_daxpy(ni, b, a, sia, c, sic);
}


template<>
const char *linalg_cblas_level1<float>::k_clazz = "cblas";

template<>
void linalg_cblas_level1<float>::add_i_i_x_x(
    void*,
    size_t ni,
    const float *a, size_t sia, float ka,
    float b, float kb,
    float *c, size_t sic,
    float d) {

    cblas_saxpy(ni, d * ka, a, sia, c, sic);
    float db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
}


template<>
void linalg_cblas_level1<float>::copy_i_i(
    void*,
    size_t ni,
    const float *a, size_t sia,
    float *c, size_t sic) {
    cblas_scopy(ni, a, sia, c, sic);
}


template<>
void linalg_cblas_level1<float>::mul1_i_x(
    void*,
    size_t ni,
    float a,
    float *c, size_t sic) {

    cblas_sscal(ni, a, c, sic);
}


template<>
float linalg_cblas_level1<float>::mul2_x_p_p(
    void*,
    size_t np,
    const float *a, size_t spa,
    const float *b, size_t spb) {

    return cblas_sdot(np, a, spa, b, spb);
}


template<>
void linalg_cblas_level1<float>::mul2_i_i_x(
    void*,
    size_t ni,
    const float *a, size_t sia,
    float b,
    float *c, size_t sic) {

    cblas_saxpy(ni, b, a, sia, c, sic);
}



} // namespace libtensor
