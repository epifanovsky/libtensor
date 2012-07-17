#include "acml_h.h"
#include "linalg_base_level2_acml.h"

namespace libtensor {


const char *linalg_base_level2_acml::k_clazz = "acml";


void linalg_base_level2_acml::i_ip_p_x(
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemv");
    dgemv('T', np, ni, d, (double*)a, sia, (double*)b, spb, 1.0, c, sic);
    stop_timer("dgemv");
}


void linalg_base_level2_acml::i_pi_p_x(
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemv");
    dgemv('N', ni, np, d, (double*)a, spa, (double*)b, spb, 1.0, c, sic);
    stop_timer("dgemv");
}


void linalg_base_level2_acml::ij_i_j_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dger");
    dger(nj, ni, d, (double*)b, sjb, (double*)a, sia, c, sic);
    stop_timer("dger");
}


void linalg_base_level2_acml::ij_ji(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    start_timer("dcopy");
    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            dcopy(nj, (double*)a + i, sja, c1, 1);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            dcopy(ni, (double*)a1, 1, c + j, sic);
        }
    }
    stop_timer("dcopy");
}


} // namespace libtensor
