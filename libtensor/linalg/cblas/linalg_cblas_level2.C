#include "cblas_h.h"
#include "linalg_cblas_level2.h"

namespace libtensor {


const char *linalg_cblas_level2::k_clazz = "cblas";


void linalg_cblas_level2::copy_ij_ji(
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


void linalg_cblas_level2::mul2_i_ip_p_x(
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b,
        spb, 1.0, c, sic);
}


void linalg_cblas_level2::mul2_i_pi_p_x(
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b,
        spb, 1.0, c, sic);
}


void linalg_cblas_level2::mul2_ij_i_j_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
}


} // namespace libtensor
