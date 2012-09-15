#include "linalg_generic_level2.h"
#include "linalg_generic_level3.h"

namespace libtensor {


const char *linalg_generic_level3::k_clazz = "generic";


void linalg_generic_level3::mul2_i_ipq_qp_x(
    void *ctx,
    size_t ni, size_t np, size_t nq,
    const double *a, size_t spa, size_t sia,
    const double *b, size_t sqb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += d * linalg_generic_level2::mul2_x_pq_qp(ctx, np, nq,
            a + i * sia, spa, b, sqb);
    }
}


void linalg_generic_level3::mul2_ij_ip_jp_x(
    void*,
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


void linalg_generic_level3::mul2_ij_ip_pj_x(
    void*,
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


void linalg_generic_level3::mul2_ij_pi_jp_x(
    void*,
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


void linalg_generic_level3::mul2_ij_pi_pj_x(
    void*,
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
