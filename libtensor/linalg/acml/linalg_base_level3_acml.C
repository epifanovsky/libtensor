#include "acml_h.h"
#include "linalg_base_level3_acml.h"

namespace libtensor {


const char *linalg_base_level3_acml::k_clazz = "acml";


void linalg_base_level3_acml::ij_ip_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    dgemm('T', 'N', nj, ni, np, d, (double*)b, sjb, (double*)a, sia,
        1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_base_level3_acml::ij_ip_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    dgemm('N', 'N', nj, ni, np, d, (double*)b, spb, (double*)a, sia,
        1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_base_level3_acml::ij_pi_jp_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    dgemm('T', 'T', nj, ni, np, d, (double*)b, sjb, (double*)a, spa,
        1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_base_level3_acml::ij_pi_pj_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    dgemm('N', 'T', nj, ni, np, d, (double*)b, spb, (double*)a, spa,
        1.0, c, sic);
    stop_timer("dgemm");
}


} // namespace libtensor
