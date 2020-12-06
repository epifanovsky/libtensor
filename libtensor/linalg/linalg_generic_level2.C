#include "linalg_generic_level2.h"

namespace libtensor {


const char linalg_generic_level2::k_clazz[] = "generic";


void linalg_generic_level2::add1_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += a[i * sia + j] * b;
    }
}


void linalg_generic_level2::add1_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] += a[j * sja + i] * b;
    }
}


void linalg_generic_level2::copy_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] = a[i * sia + j] * b;
    }
}


void linalg_generic_level2::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i];
    }
}


void linalg_generic_level2::copy_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i] * b;
    }
}


void linalg_generic_level2::mul2_i_ip_p_x(
    void*,
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


void linalg_generic_level2::mul2_i_pi_p_x(
    void*,
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


void linalg_generic_level2::mul2_ij_i_j_x(
    void*,
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


double linalg_generic_level2::mul2_x_pq_pq(
    void*,
    size_t np, size_t nq,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    double c = 0.0;
    for(size_t p = 0; p < np; p++) {
        const double *a1 = a + p * spa, *b1 = b + p * spb;
        for(size_t q = 0; q < nq; q++) {
            c += a1[q] * b1[q];
        }
    }
    return c;
}


double linalg_generic_level2::mul2_x_pq_qp(
    void*,
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
