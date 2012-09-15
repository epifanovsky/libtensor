#include "essl_h.h"
#include "linalg_essl_level1.h"

namespace libtensor {


const char *linalg_essl_level1::k_clazz = "essl";


void linalg_essl_level1::add_i_i_x_x(
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    daxpy(ni, d * ka, (double*)a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
}


void linalg_essl_level1::i_x(
    size_t ni,
    double a,
    double *c, size_t sic) {

    dscal(ni, a, c, sic);
}


double linalg_essl_level1::x_p_p(
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    return ddot(np, (double*)a, spa, (double*)b, spb);
}


void linalg_essl_level1::i_i_x(
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    daxpy(ni, b, (double*)a, sia, c, sic);
}


} // namespace libtensor
