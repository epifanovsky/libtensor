#include "linalg_generic_level1.h"

namespace libtensor {


const char *linalg_generic_level1::k_clazz = "generic";


void linalg_generic_level1::add_i_i_x_x(
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += (ka * a[i * sia] + kb * b) * d;
    }
}


void linalg_generic_level1::copy_i_i(
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) c[i * sic] = a[i * sia];
}


void linalg_generic_level1::mul1_i_x(
    size_t ni,
    double a,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) c[i * sic] *= a;
}


double linalg_generic_level1::mul2_x_p_p(
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    double c = 0.0;
    for(size_t p = 0; p < np; p++) c += a[p * spa] * b[p * spb];
    return c;
}


void linalg_generic_level1::mul2_i_i_x(
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b;
}


void linalg_generic_level1::mul2_i_i_i_x(
    size_t ni,
    const double *a, size_t sia,
    const double *b, size_t sib,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++) c[i * sic] += d * a[i * sia] * b[i * sib];
}


} // namespace libtensor
