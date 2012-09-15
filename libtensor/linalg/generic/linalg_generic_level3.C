#include "linalg_generic_level2.h"
#include "linalg_generic_level3.h"

namespace libtensor {


void linalg_generic_level3::i_ipq_qp_x(
    size_t ni, size_t np, size_t nq,
    const double *a, size_t spa, size_t sia,
    const double *b, size_t sqb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += d * linalg_generic_level2::x_pq_qp(
            np, nq, a + i * sia, spa, b, sqb);
    }
}


void linalg_generic_level3::ij_ip_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        double cij = 0.0;
        for(size_t p = 0; p < np; p++) {
            cij += a[i * sia + p] * b[j * sjb + p];
        }
        c[i * sic + j] += d * cij;
    }
}


void linalg_generic_level3::ij_ip_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++)
    for(size_t p = 0; p < np; p++) {
        double aip = a[i * sia + p];
        for(size_t j = 0; j < nj; j++) {
            c[i * sic + j] += d * aip * b[p * spb + j];
        }
    }
}


void linalg_generic_level3::ij_pi_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++)
    for(size_t p = 0; p < np; p++) {
        c[i * sic + j] += d * a[p * spa + i] * b[j * sjb + p];
    }
}


void linalg_generic_level3::ij_pi_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    for(size_t p = 0; p < np; p++)
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += d * a[p * spa + i] * b[p * spb + j];
    }
}


} // namespace libtensor
