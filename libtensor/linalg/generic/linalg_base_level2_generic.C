#include "linalg_base_level2_generic.h"

namespace libtensor {


void linalg_base_level2_generic::add1_ij_ij_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += a[i * sia + j] * b;
    }
}


void linalg_base_level2_generic::add1_ij_ji_x(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] += a[j * sja + i] * b;
    }
}


void linalg_base_level2_generic::i_ip_p_x(
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++) {
        double ci = 0.0;
        for(size_t p = 0; p < np; p++) {
            ci += a[i * sia + p] * b[p * spb];
        }
        c[i * sic] += d * ci;
    }
}


void linalg_base_level2_generic::i_pi_p_x(
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    for(size_t p = 0; p < np; p++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += d * a[p * spa + i] * b[p * spb];
    }
}


void linalg_base_level2_generic::ij_i_j_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += d * a[i * sia] * b[j * sjb];
    }
}


void linalg_base_level2_generic::ij_ji(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i];
    }
}


void linalg_base_level2_generic::ij_ij_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] = a[i * sia + j] * b;
    }
}


void linalg_base_level2_generic::ij_ji_x(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i] * b;
    }
}


double linalg_base_level2_generic::x_pq_qp(
    size_t np, size_t nq,
    const double *a, size_t spa,
    const double *b, size_t sqb) {

    double c = 0.0;
    for(size_t p = 0; p < np; p++)
    for(size_t q = 0; q < nq; q++) {
        c += a[p * spa + q] * b[q * sqb + p];
    }
    return c;
}


} // namespace libtensor
