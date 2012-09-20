#include "cblas_h.h"
#include "linalg_cblas_level1.h"

namespace libtensor {


const char *linalg_cblas_level1::k_clazz = "cblas";


void linalg_cblas_level1::add_i_i_x_x(
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


void linalg_cblas_level1::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    cblas_dcopy(ni, a, sia, c, sic);
}


void linalg_cblas_level1::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    cblas_dscal(ni, a, c, sic);
}


double linalg_cblas_level1::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    return cblas_ddot(np, a, spa, b, spb);
}


void linalg_cblas_level1::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    cblas_daxpy(ni, b, a, sia, c, sic);
}


} // namespace libtensor
