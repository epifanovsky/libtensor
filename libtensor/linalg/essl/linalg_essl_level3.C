#include "essl_h.h"
#include "linalg_essl_level3.h"

namespace libtensor {


const char *linalg_essl_level3::k_clazz = "essl";


void linalg_essl_level3::ij_ip_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    dgemm("T", "N", nj, ni, np, d, (double*)b, sjb, (double*)a, sia,
        1.0, c, sic);
}


void linalg_essl_level3::ij_ip_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    dgemm("N", "N", nj, ni, np, d, (double*)b, spb, (double*)a, sia,
        1.0, c, sic);
}


void linalg_essl_level3::ij_pi_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    dgemm("T", "T", nj, ni, np, d, (double*)b, sjb, (double*)a, spa,
        1.0, c, sic);
}


void linalg_essl_level3::ij_pi_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    dgemm("N", "T", nj, ni, np, d, (double*)b, spb, (double*)a, spa,
        1.0, c, sic);
}


} // namespace libtensor
